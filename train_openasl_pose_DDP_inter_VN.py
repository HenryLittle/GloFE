import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import os
import pickle
import sys
import argparse
import json
import random
import shutil
import math


from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Tuple, Optional, Union
from tensorboardX import SummaryWriter

# Project related
from models.trans_model_inter_vn import TransBaseModel
from models.ctrgcn_base_p76 import Model as PoseBackbone
from utils.beam_search import AutoRegressiveBeamSearch
from utils.easydict import EasyDict as edict
from utils.mutils import generate_beam, compute_bleu, quick_bleu_metric

# DDP
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel



class OpenASLPoseDataset(Dataset):
    def __init__(self, arg_dict, tokenizer, phase='train', split='train'):
        self.tokenizer = tokenizer
        self.phase = phase

        # path to openasl
        self.feat_path = arg_dict.get('feat_path', None)
        self.label_path = arg_dict.get('label_path', None)
        assert self.feat_path is not None and self.feat_path is not None

        self.local_rank = arg_dict.get('local_rank', 0)
        self.eos_token = arg_dict.get('eos_token', '.')
        # information about input clips (features)
        self.visual_token_num = arg_dict.get('clip_length', 16)
        self.visual_token_dim = arg_dict.get('prefix_dim', 2048)

        
        # label_path = os.path.join(f'/mnt/workspace/openasl-pre/openasl-v1.0.tsv')
        assert self.label_path is not None, 'Specify --label_path'
        data_frame = pd.read_csv(self.label_path, sep='\t')

        # select by split ['train', 'valid']
        data_frame = data_frame.loc[data_frame['split'].str.contains(split)]

        def filter_missing(row):
            full_path = os.path.join(self.feat_path, f'{row["vid"]}.pkl')
            return os.path.exists(full_path) and os.path.getsize(full_path) > 0

        is_missing = data_frame.apply(filter_missing, axis=1)
        df_filtered = data_frame[is_missing]
        if self.local_rank == 0:
            print(f'Split:{split}\nBefore filtering: {len(data_frame)}\n After filtering: {len(df_filtered)}')
        # translation labels and sample names (split agnostic)
        self.translation = df_filtered['raw-text'].to_list()
        self.video_names = df_filtered['vid'].to_list()
        
        self.vn_vocab = 5523
        self.matched_VNs = json.load(open('notebooks/openasl-v1.0/uncased_filtred_glove_VN_matched_train.json', 'r'))
        self.vn_to_idx = {}
        with open('notebooks/openasl-v1.0/uncased_filtred_glove_VN_idxs.txt', 'r') as f:
            content = f.readlines()
            for line in content:
                items = line.strip().split(' ')
                self.vn_to_idx[items[1]] = int(items[0])
        vn_lens = [len(v) for _,v in self.matched_VNs.items()]
        self.max_vns = max(vn_lens)

        self.translation_token_ids = []  # encoded indices
        for trans in self.translation:
            # add eos token to labels
            trans_ids = self.tokenizer.encode('<s>') + self.tokenizer.encode(trans) + self.tokenizer.encode(self.eos_token)
            self.translation_token_ids.append(torch.tensor(trans_ids, dtype=torch.int64))
        
        assert len(self.translation_token_ids) == len(
            self.video_names), f'Text ids count:{len(self.translation_token_ids)}\tVid count:{len(self.video_names)}'
        all_len = torch.tensor([len(tk)
                               for tk in self.translation_token_ids]).float()
        self.max_seq_len = min(
            int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
        if self.local_rank == 0:
            print(f'Max sequence length:{self.max_seq_len}')

    def __len__(self):
        return len(self.video_names)

    def read_pose_files(self, index: int):
        # MMPose 76
        body_sample_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        face_sample_indices = [71, 77, 85, 89] + \
                              [40, 42, 44, 45, 47, 49] + \
                              [59, 60, 61, 62, 63, 64] + [65, 66, 67, 68, 69, 70] + \
                              [50]
        
        # read files
        vid_name = self.video_names[index]

        file_path = os.path.join(self.feat_path, f'{vid_name}.pkl')
        with open(file_path, 'rb') as f:
            pose_keypoints = pickle.load(f)  # T K(133) C

        # 23(17+6) 11 selected
        body_pose = pose_keypoints[:, body_sample_indices, :]
        hand_right = pose_keypoints[:, 91:112, :]  # 21 Keypoints
        hand_left = pose_keypoints[:, 112:, :]  # 21 Keypoints
        face = pose_keypoints[:, face_sample_indices, :]  # 23 Keypoints

        pose_tuple = (body_pose, hand_left, hand_right, face)
        pose_cated = np.concatenate(
            pose_tuple, axis=1)  # [F, 11+21+21+23=76, 3]

        # scale to [-1, 1]
        # normalization, same as pre-training, might not align with actual output shape which assumed to be [288, 384]
        pose_cated[:, :, 0:2] = 2.0 * ((pose_cated[:, :, 0:2] / 256.0) - 0.5)

        # pad pose
        T, V, C = pose_cated.shape
        # assert T == len(filenames)
        if T < self.visual_token_num:
            diff = self.visual_token_num - T
            pose_output = np.concatenate(
                (pose_cated, np.zeros((diff, V, C))), axis=0)
        elif T > self.visual_token_num:
            if self.phase == 'train':
                diff = T - self.visual_token_num
                offset = np.random.randint(0, diff)
                pose_output = pose_cated[offset: offset +
                                     self.visual_token_num, :, :]
            elif self.phase == 'test':
                offset = 0
                pose_output = pose_cated[offset: offset +
                                     self.visual_token_num, :, :]
        else:
            pose_output = pose_cated

        pose_length = T if T <= self.visual_token_num else self.visual_token_num

        return pose_output, pose_length

    def rand_view_transform(self, X, agx, agy, s):
        if X.shape[-1] == 2:
            padding = np.zeros((X.shape[0], X.shape[1], 1))
            X = np.concatenate((X, padding), axis=2)
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1,              0,             0],
                         [0,  math.cos(agx), math.sin(agx)],
                         [0, -math.sin(agx), math.cos(agx)]])

        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)],
                         [0, 1,              0],
                         [math.sin(agy), 0,  math.cos(agy)]])

        Ss = np.asarray([[s, 0, 0],
                         [0, s, 0],
                         [0, 0, s]])

        X0 = np.dot(np.reshape(X, (-1, 3)), np.dot(Ry, np.dot(Rx, Ss)))
        X = np.reshape(X0, X.shape)
        return X

    def normalize_joints(self, value):
        T, V, C = value.shape
        # scale to [-1, 1]
        scalerValue = np.reshape(value, (-1, C))
        scalerValue = (scalerValue - np.min(scalerValue, axis=0)) / \
            ((np.max(scalerValue, axis=0) - np.min(scalerValue, axis=0)) + 1e-5)

        scalerValue = scalerValue * 2 - 1
        scalerValue = np.reshape(scalerValue, (-1, V, C))

        return scalerValue

    def pad_token_ids(self, index: int, pad_const=1):
        tokens = self.translation_token_ids[index]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat(
                (tokens, torch.ones(padding, dtype=torch.int64) * -1))
            self.translation_token_ids[index] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.translation_token_ids[index] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens_length = mask.sum()
        tokens[~mask] = pad_const
        mask = mask.float()

        return tokens, mask, tokens_length
    
    def get_vn(self, index: int):
        vid = self.video_names[index]
        vns = self.matched_VNs[vid]
        vn_idxs = [self.vn_to_idx[x] for x in vns]
        vn_idxs =  torch.tensor(vn_idxs, dtype=torch.int64)
        vn_len = len(vn_idxs)
        pad_len = self.max_vns - vn_len
        if pad_len > 0:
            vn_idxs = torch.cat((vn_idxs, torch.ones(pad_len, dtype=torch.int64)))
        return vn_idxs, vn_len

    def __getitem__(self, index: int):
        if self.phase == 'train':
            text_tokens, mask, token_length = self.pad_token_ids(
                index)  # [max_seq_len]
            visual_prefix, visual_length = self.read_pose_files(index)
            agx = np.random.randint(-60, 60)
            agy = np.random.randint(-60, 60)
            s = np.random.uniform(0.5, 1.5)
            # augmentation
            visual_prefix[:, :, :2] = self.rand_view_transform(
                visual_prefix[:, :, :2], agx, agy, s)[:, :, :2]
            visual_prefix[:, :, :2] = self.normalize_joints(
                visual_prefix[:, :, :2])
            # reorder [T V C] -> [C T V]
            visual_prefix = np.transpose(visual_prefix, (2, 0, 1))
            visual_prefix = torch.from_numpy(visual_prefix)
            visual_prefix = visual_prefix.type(torch.FloatTensor)
            vn_idxs, vn_len = self.get_vn(index)
            return text_tokens, mask, visual_prefix, token_length, visual_length, vn_idxs, vn_len
        elif self.phase == 'test':
            visual_prefix, visual_length = self.read_pose_files(index)
            visual_prefix[:, :, :2] = self.normalize_joints(
                visual_prefix[:, :, :2])
            # reorder [T V C] -> [C T V]
            visual_prefix = np.transpose(visual_prefix, (2, 0, 1))
            visual_prefix = torch.from_numpy(visual_prefix)
            visual_prefix = visual_prefix.type(torch.FloatTensor)
            return visual_prefix, index, visual_length


def save_config(args: argparse.Namespace, output_path: str):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(output_path, f"exp_config.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def init_logging(output_dir, reuse=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Tensorboard logging
    tb_log_dir = os.path.join(output_dir, "tb_logs")
    if os.path.isdir(tb_log_dir) and not reuse:
        print('Dir existed: ', tb_log_dir)
        answer = input('Delete it? y/n:')
        if answer == 'y':
            shutil.rmtree(tb_log_dir)
            print('Dir removed: ', tb_log_dir)
            input('Refresh the website of tensorboard by pressing any keys')
        else:
            print('Dir not removed: ', tb_log_dir)
    train_writer = SummaryWriter(os.path.join(tb_log_dir, 'train'), 'train')
    return train_writer


def train(datasets, model, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):

    device = torch.device(f'cuda:{args.local_rank}')
    batch_size = args.bs
    epochs = args.epochs
    label_smoothing = args.ls
    best_b4 = 0  # best running metric
    if args.local_rank == 0:
        train_writer = init_logging(
            output_dir=output_dir, reuse=(args.resume != -1))
        save_config(args, output_dir)

    # Init model
    # model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr, no_deprecation_warning=True)
    dataset_train, dataset_dev, dataset_test = datasets
    if args.ngpus > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train, num_replicas=args.ngpus, rank=args.local_rank)
    else:
        sampler = None
    train_dataloader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=(sampler is None),
        drop_last=True,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        # prefetch_factor=4
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs *
        len(train_dataloader)
    )

    if args.resume != -1:
        epoch_st = args.resume + 1
        assert epoch_st < epochs
        global_step = args.resume * len(train_dataloader)
    else:
        global_step = 0
        epoch_st = 0

    for epoch in range(epoch_st, epochs):
        if args.local_rank == 0:
            print(f">>> Running exp: {args.work_dir}, Prefix: {args.prefix}")
            train_writer.add_scalar('epoch', epoch, global_step)
            sys.stdout.flush()
            progress = tqdm(total=len(train_dataloader),
                            desc=f'Epoch {epoch:03d}')
        # Run for One Epoch
        for idx, (tokens, mask, prefix, token_length, visual_length, vn_idxs, vn_len) in enumerate(train_dataloader):
            if args.local_rank == 0:
                train_writer.add_scalar(
                    'lr', scheduler.optimizer.param_groups[0]['lr'], global_step)
            model.zero_grad()
            tokens, mask, prefix, token_length, visual_length = tokens.to(device), mask.to(device), prefix.to(
                device, dtype=torch.float32), token_length.to(device), visual_length.to(device)
            vn_idxs, vn_len = vn_idxs.to(device), vn_len.to(device)
            outputs = model(
                x=prefix,
                x_length=visual_length,
                tgt=tokens,
                tgt_length=token_length,
                vn_idxs=vn_idxs,
                vn_len=vn_len
            )
            # Back prop and Reseting grads
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if args.local_rank == 0:
                progress.set_postfix({"loss": loss.item()})
                progress.update()
                train_writer.add_scalar('loss', loss.item(), global_step)
                train_writer.add_scalar('inter_loss', outputs['inter_cl'].item(), global_step)

                if (idx + 1) % 1000 == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                    )
            global_step += 1
        if args.local_rank == 0:
            # End of Epoch
            progress.close()

        # Evaluate
        is_best = False
        if (epoch > -1):
            if args.ngpus > 1:
                eval_model = model.module
            else:
                eval_model = model

            save_results = (epoch % args.save_every ==
                            0 or epoch == epochs - 1)
            # evaluate and save result
            dev_bleu = eval(dataset_dev, eval_model, args, device, split='valid', 
                            save_results=save_results, output_dir=output_dir, output_prefix=f'{epoch:03d}-valid')
            test_bleu = eval(dataset_test, eval_model, args, device, split='test',
                            save_results=save_results, output_dir=output_dir, output_prefix=f'{epoch:03d}-test')

            if args.local_rank == 0:
                train_writer.add_scalar('valid/BLEU-1', dev_bleu[0], epoch)
                train_writer.add_scalar('valid/BLEU-2', dev_bleu[1], epoch)
                train_writer.add_scalar('valid/BLEU-3', dev_bleu[2], epoch)
                train_writer.add_scalar('valid/BLEU-4', dev_bleu[3], epoch)
                train_writer.add_scalar('test/BLEU-1', test_bleu[0], epoch)
                train_writer.add_scalar('test/BLEU-2', test_bleu[1], epoch)
                train_writer.add_scalar('test/BLEU-3', test_bleu[2], epoch)
                train_writer.add_scalar('test/BLEU-4', test_bleu[3], epoch)

                if dev_bleu[3] > best_b4:
                    is_best = True
                    best_b4 = dev_bleu[3]

            model.train()
        # Save ckpt periodically | best metric
        if args.local_rank == 0:
            # Save checkpoints
            if is_best:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        output_dir, f"best-{best_b4:.4f}-at-{epoch:03d}.pt"),
                )
            elif (epoch % args.save_every == 0 or epoch == epochs - 1) and (args.epochs - epoch < 15):
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        output_dir, f"{output_prefix}-{epoch:03d}.pt"),
                )
        # End of one epoch
    # End of tarining
    return model


def text_to_word_token(text_list, eos_token='.'):
    # remove whitespaces and split words by ' '
    # NOTE: this does not remove special tokens
    return [re.sub(r'[ \n]+', ' ', t.strip().replace(eos_token, '')).split(' ') for t in text_list]


def eval(dataset, model, args, device, split, save_results: bool = False,
         output_dir: str = ".", output_prefix: str = ""):
    model.eval()
    tokenizer = dataset.tokenizer
    eval_dataloader = DataLoader(
        dataset, batch_size=args.bs, shuffle=False, drop_last=False, num_workers=8)

    gen_text_list = []
    ref_text_list = []
    if args.local_rank == 0:
        progress = tqdm(total=len(eval_dataloader), desc=output_prefix)
    with torch.no_grad():
        for idx, (visual_prefix, text_label_index, visual_length) in enumerate(eval_dataloader):
            B = visual_prefix.shape[0]
            visual_prefix, visual_length = visual_prefix.to(
                device), visual_length.to(device)

            # load reference text
            for b in range(B):
                label_text = dataset.translation[text_label_index[b].item()]
                ref_text_list.append(label_text.lower())
            
            # generate hyposis
            predicted_token_ids = model(
                x=visual_prefix,
                x_length=visual_length,
                phase='test',
            )

            hyposis_list = tokenizer.batch_decode(
                predicted_token_ids, skip_special_tokens=True)
            gen_text_list.extend(hyposis_list)

            if args.local_rank == 0:
                progress.update()
            
    if args.local_rank == 0:
        progress.close()

    # use tokenizer to convert text to tokens, may need to use space instead
    ref_text_tokens = text_to_word_token(ref_text_list, dataset.eos_token)
    gen_text_tokens = text_to_word_token(gen_text_list, dataset.eos_token)
    # wrap ref in List[] to match input requirements of comput_bleu
    ref_tokens_list = [[t] for t in ref_text_tokens]
    bleu_results = quick_bleu_metric(ref_tokens_list, gen_text_tokens, split)
    if args.local_rank == 0 and save_results:
        with open(os.path.join(output_dir, f"{output_prefix}_{split}_eval.csv"), 'w') as f:
            for gen_text, label_text in zip(gen_text_list, ref_text_list):
                f.write(f"{gen_text}|{label_text}\n")
        with open(os.path.join(output_dir, f"{output_prefix}_{split}_eval_tokens.pkl"), 'wb') as f:
            pickle.dump((gen_text_tokens, ref_text_tokens), f)
    return bleu_results


def construct_model(model_cls, args, distributed=False):
    rank = args.local_rank
    generator = AutoRegressiveBeamSearch(
        eos_index=2,
        max_steps=args.max_gen_tks,
        beam_size=args.num_beams,
        per_node_beam_size=2,
    )
    model = model_cls(args, generator=generator, sos_index=0)
    # move model to GPU
    device = torch.device(f'cuda:{rank}')
    model.to(device)
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True)
    return model


def init_random_seeds(random_seed=0, rank=0):
    # eliminate isomophisim across ranks
    the_seed = random_seed + rank
    torch.manual_seed(the_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(the_seed)
    np.random.seed(the_seed)
    random.seed(the_seed)
    # These two slows down training
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--work_dir_prefix', default='/mnt/workspace/slt_baseline/work_dir', help='path to work_dir')
    parser.add_argument('--work_dir', default='checkpoints',
                        help='<dataset>/<exp_name>')
    parser.add_argument(
        '--prefix', type=str, default='phoenix_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--warm_up', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--phase', type=str, default='train',
                        choices=['train', 'test'])
    parser.add_argument('--split', type=str, default='dev',
                        choices=['train', 'dev', 'val', 'test', 'valid'])
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--resume', type=int, default=-
                        1, help='epoch to resume from')
    # DDP related
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ngpus', type=int, default=1,
                        help='number of gpus used, equivilent to world_size(local)')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='pass rank info throughout the script, DO NOT input through command line')
    # Datasets
    parser.add_argument('--feat_path', type=str,
                        help='path to the OpenASL folder')
    parser.add_argument('--label_path', type=str,
                        help='path to OpenASL\'s .csv label file')
    parser.add_argument('--clip_length', type=int, default=10,
                        help='number of input visual tokens')
    parser.add_argument('--tokenizer', type=str,
                        default='/mnt/workspace/slt_baseline/notebooks/openasl-bpe25000-tokenizer')
    parser.add_argument('--eos_token', type=str, default='.',
                        help='eos token for text generation')

    # Generator config
    parser.add_argument('--max_gen_tks', type=int, default=35,
                        help='max generated token number for decoder')
    parser.add_argument('--num_beams', type=int, default=5,
                        help='number of beams for beam search during inference')

    # Model related arguments
    parser = TransBaseModel.add_args(parser)  # add model related arguments
    args = parser.parse_args()

    model_cls = TransBaseModel

    # [Steup]
    if args.ngpus > 1:
        # init DDP
        distributed = True
        dist.init_process_group(backend='nccl')
        args.local_rank = dist.get_rank()
        torch.cuda.set_device(args.local_rank)
    else:
        # on process, treat as the master process
        args.local_rank = 0
        distributed = False

    # Ensure each process has the same initialization
    init_random_seeds(args.seed, args.local_rank)
    print('[RANK] running on rank:', args.local_rank)
    if args.local_rank == 0:
        for k, v in vars(args).items():
            print(f'{k}'.rjust(18), f'\t{v}')

    output_dir = os.path.join(args.work_dir_prefix, args.work_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True)

    # [Load model weights]
    if args.weights and os.path.exists(args.weights):
        if args.local_rank == 0:
            print(f"Loaded weights from: {args.weights}")
        weights = torch.load(args.weights)
    else:
        weights = None

    # [Lauch running process]
    if args.phase == 'train':
        # [Init datasets]
        dataset = OpenASLPoseDataset(vars(args), tokenizer=tokenizer, phase='train', split='train')
        dataset_dev = OpenASLPoseDataset(vars(args), tokenizer=tokenizer, phase='test', split='valid')
        dataset_test = OpenASLPoseDataset(vars(args), tokenizer=tokenizer, phase='test', split='test')

        # pass the namespace as a dict
        model = construct_model(model_cls, args, distributed)
        if weights:
            model.load_state_dict(weights)

        train((dataset, dataset_dev, dataset_test), model, args, output_dir=output_dir,
              output_prefix=args.prefix, warmup_steps=args.warm_up, lr=args.lr)
    elif args.phase == 'test':
        assert weights is not None, f'{args.weights} dose not exist'
        # load experiment config json file
        if os.path.exists(os.path.join(output_dir, f'phoenix_prefix.json')):
            json_path = os.path.join(output_dir, f'phoenix_prefix.json')
        else:
            json_path = os.path.join(output_dir, f"exp_config.json")
        config = json.load(open(json_path, 'r'))
        # construct model
        model = construct_model(model_cls, edict(config), distributed)

        # >>>>>> HACK
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in weights.items():
            # remove 'module.' of DataParallel/DistributedDataParallel
            name = k[7:]
            new_state_dict[name] = v
        weights = new_state_dict
        # >>>>>>

        model.load_state_dict(weights, strict=False)
        
        dataset_valid = OpenASLPoseDataset(
            config, tokenizer=tokenizer, phase='test', split='valid')
        dataset_test = OpenASLPoseDataset(
            config, tokenizer=tokenizer, phase='test', split='test')
        device = torch.device(f'cuda:{args.local_rank}')
        eval(dataset_valid, model, args, device, 'valid', save_results=True,
             output_dir=output_dir, output_prefix=f'{args.prefix}-valid')
        eval(dataset_test, model, args, device, 'test', save_results=True,
             output_dir=output_dir, output_prefix=f'{args.prefix}-test')


if __name__ == '__main__':
    # disable tokenizer parallelism and prevent warning when using dataloader
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # to load tensor in __getitem__ when workers are used
    # [UPDATE] don't know why this cause mmap to misbehave in a new envirionment, remove for now
    # torch.multiprocessing.set_start_method('spawn')
    main()
