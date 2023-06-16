import sys
import torch
import pandas as pd
import numpy as np
import pickle
import math
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.append("..")
# from models.ctrgcn_base_p76 import Model as PoseBackbone
from models.ctrgcn_base import Model as PoseBackbone

class PoseBackboneWrapper(nn.Module):
    def __init__(self):
        super(PoseBackboneWrapper, self).__init__()
        self.pose_model = PoseBackbone(
                num_class=2000, num_point=78, num_person=1, 
                graph='models.graph.openpose_78.Graph',
                graph_args={'labeling_mode': 'spatial'}, drop_out=0)
        pose_weights = torch.load('/mnt/workspace/slt_baseline/models/ckpt/ctr_op78_mix_HF05_F64_e1/runs-82-93316.pt')
        # output [B C T V]
        self.pose_model.load_state_dict(pose_weights, strict=False)
    
    def forward(self, prefix):
        pose_output = self.pose_model(prefix) # B C T V
        pose_pool = pose_output.mean(-1) # B C T
        prefix = pose_pool.transpose(-1, -2) # B T C
        out_cls = self.pose_model.fc(prefix) # B T Class
        return out_cls

class Phoenix2014TDataset(Dataset):
    def __init__(self, arg_dict, phoenix_root: str, tokenizer, phase='train', split='train'):
        self.tokenizer = tokenizer
        self.phase = phase

        self.feat_path = arg_dict.get('feat_path') # path to 'openpose_output/json'
        self.eos_token = arg_dict.get('eos_token', '.')
        self.prefix_length = arg_dict.get('prefix_length', 16) 
        self.normalize_prefix = arg_dict.get('normalize_prefix', False) 
        self.lm = arg_dict.get('lm', 'gpt') 
        self.mbart_lang = arg_dict.get('mbart_lang', 'de_DE')  # language code for mbart
        self.local_rank = arg_dict.get('local_rank', 0)
        # information about input clips
        self.visual_token_num =  arg_dict.get('clip_length', 16)  
        self.visual_token_dim = 2048
        label_path = os.path.join(phoenix_root, f'/mnt/workspace/How2Sign/how2sign_realigned_{split}.csv')
        data_frame = pd.read_csv(label_path, sep='\t')
        # translation labels and sample names (split agnostic)
        self.translation = list(data_frame['SENTENCE'])
        self.video_names = list(data_frame['SENTENCE_NAME'])

        # filter out missing parts
        with open('/mnt/workspace/How2Sign/tools/how2sign_missing.txt', 'r') as f:
            names = f.readlines()
            missing = [x.strip() for x in names]
        vid_filtered, trans_filtered = [], []
        for vid, trans in zip(self.video_names, self.translation):
            if vid not in missing:
                vid_filtered.append(vid)
                trans_filtered.append(trans)
        if self.local_rank == 0:
            print('Before filtering:', len(self.video_names), '\nAfter filtering:', len(vid_filtered), '\n')
        self.video_names = vid_filtered
        self.translation = trans_filtered
        # load all features to memory
        self.cache_path = '/mnt/workspace/How2Sign/openpose_output/cache_inst'
        if os.path.exists(self.cache_path):
            cache_length = len(os.listdir(self.cache_path))
            if self.local_rank == 0:
                print('Using cache from:', self.cache_path)
                print('Total items:', cache_length)
            self.use_cache = True
        else:
            self.use_cache = False


        self.translation_token_ids = [] # encoded indices
        for trans in self.translation:
            # add eos token to labels
            if self.lm == 'gpt':
                trans_ids = self.tokenizer.encode(trans) + self.tokenizer.encode(self.eos_token)
            elif self.lm == 'mbart':
                # target labels
                # trans_ids = self.tokenizer.encode(f'{mbart_lang} </s> {trans}. </s>', add_special_tokens=False)
                # trans_ids = self.tokenizer.encode(f'{mbart_lang} {trans}. </s>', add_special_tokens=False)
                # trans_ids = self.tokenizer.encode(f'{trans} </s> {mbart_lang}', add_special_tokens=False)
                trans_ids = self.tokenizer.encode(f'{mbart_lang} {trans} </s>', add_special_tokens=False)
            else: 
                raise NotImplementedError
            self.translation_token_ids.append(torch.tensor(trans_ids, dtype=torch.int64))
        all_len = torch.tensor([len(tk) for tk in self.translation_token_ids]).float()
        # not sure the origin of this method
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

    def __len__(self):
        return len(self.video_names)

    def read_pose_files(self, index: int):
        # read and filter openpose outputs, to match MP76 (2 extra upper body is removed)
        # returen pose shape [MaxN, V, 3] np
        # body_pose_exclude = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21, 1, 8}
        body_pose_exclude = {9, 10, 11, 22, 23, 24, 12, 13, 14, 19, 20, 21}
        body_sample_indices = [x for x in range(25) if x not in body_pose_exclude]
        #               0, 1, 2,  3, 4, 5, 6, 7, 8, 9, 10
        # body_op_2_mp = [0, 8, 7, 10, 9, 4, 1, 5, 2, 6, 3]
        face_sample_indices = [71, 77, 85, 89] + \
                              [40, 42, 44, 45, 47, 49] + \
                              [59, 60, 61, 62, 63, 64] + [65, 66, 67, 68, 69, 70] + \
                              [50]
        face_sample_indices = [x - 23 for x in face_sample_indices]
        # read files
        vid_name = self.video_names[index]
        if self.use_cache:
            with open(os.path.join(self.cache_path, f'{vid_name}.pkl'), 'rb') as f:
                joints = pickle.load(f)

            body_pose = joints[:, :25, :]
            face = joints[:, 67: , :]

            body_pose = body_pose[:, body_sample_indices, :]
            # body_pose = body_pose[:, body_op_2_mp, :] # reorder, map op points to mp
            face = face[:, face_sample_indices, :]

            pose_tuple = (body_pose, joints[:, 25 : 67, :], face)
            pose_cated = np.concatenate(pose_tuple, axis=1) 
        else:
            file_path = os.path.join(self.feat_path, vid_name)
            filenames = sorted(os.listdir(file_path))
            pose_list = []
            for name in filenames:
                with open(os.path.join(file_path, name), 'r') as f:
                    content = json.load(f)
                pose_dict = content['people'][0]
                body_pose = np.array(pose_dict['pose_keypoints_2d']).reshape(-1, 3)        # 25 -> 11
                hand_left = np.array(pose_dict['hand_left_keypoints_2d']).reshape(-1, 3)   # 21
                hand_right = np.array(pose_dict['hand_right_keypoints_2d']).reshape(-1, 3) # 21
                face = np.array(pose_dict['face_keypoints_2d']).reshape(-1, 3)             # 70 -> 23
                # sample certain keypoints
                body_pose = body_pose[body_sample_indices, :]
                face = face[face_sample_indices, :]

                pose_tuple = (body_pose, hand_left, hand_right, face)
                pose_frame_cated = np.concatenate(pose_tuple, axis=0) 
                pose_list.append(pose_frame_cated)
            pose_cated = np.stack(pose_list, axis=0) # [T, V, 3]
            # scale to [-1, 1]
            pose_cated[:, :, 0:2] = 2.0 * ((pose_cated[:, :, 0:2] / [1280.0, 720.0]) - 0.5) 
        # pad pose 
        # T, V, C = pose_cated.shape
        # # assert T == len(filenames)
        # if T < self.visual_token_num:
        #     diff = self.visual_token_num - T
        #     pose_output = np.concatenate((pose_cated, np.zeros((diff, V, C))), axis=0)
        # elif T > self.visual_token_num:
        #     diff = T - self.visual_token_num
        #     offset = np.random.randint(0, diff)
        #     pose_output = pose_cated[offset : offset + self.visual_token_num, :, :]
        # else:
        pose_output = pose_cated
        
        return pose_output
    
    def rand_view_transform(self,X, agx, agy, s):
        if X.shape[-1] == 2:
            padding = np.zeros((X.shape[0], X.shape[1], 1))
            X = np.concatenate((X, padding), axis=2)
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1,              0,             0], 
                         [0,  math.cos(agx), math.sin(agx)], 
                         [0, -math.sin(agx), math.cos(agx)]])

        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], 
                         [            0, 1,              0],
                         [math.sin(agy), 0,  math.cos(agy)]])

        Ss = np.asarray([[s, 0, 0],
                         [0, s, 0],
                         [0, 0, s]])

        X0 = np.dot(np.reshape(X,(-1,3)), np.dot(Ry,np.dot(Rx,Ss)))
        X = np.reshape(X0, X.shape)
        return X

    def normalize_joints(self, value):
        T, V, C = value.shape
        # scale to [-1, 1]
        scalerValue = np.reshape(value, (-1, C))
        scalerValue = (scalerValue - np.min(scalerValue, axis=0)) / ((np.max(scalerValue, axis=0) - np.min(scalerValue, axis=0)) + 1e-5)
        
        scalerValue = scalerValue * 2 - 1
        scalerValue = np.reshape(scalerValue, (-1, V, C))

        return scalerValue
    
    def pad_token_ids(self, index: int):
        tokens = self.translation_token_ids[index]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.ones(padding, dtype=torch.int64) * -100))
            self.translation_token_ids[index] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.translation_token_ids[index] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        if self.lm == 'gpt':
            mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask


    def __getitem__(self, index: int):
        if self.phase == 'train':
            text_tokens, mask = self.pad_token_ids(index) # [max_seq_len]
            visual_prefix = self.read_pose_files(index)
            agx = np.random.randint(-60, 60)
            agy = np.random.randint(-60, 60)
            s = np.random.uniform(0.5, 1.5)  
            # augmentation
            visual_prefix[:, :, :2] = self.rand_view_transform(visual_prefix[:, :, :2], agx, agy, s)[:, :, :2]
            visual_prefix[:, :, :2] = self.normalize_joints(visual_prefix[:, :, :2])
            # reorder [T V C] -> [C T V]
            visual_prefix = np.transpose(visual_prefix, (2, 0, 1))
            visual_prefix = torch.from_numpy(visual_prefix)
            visual_prefix = visual_prefix.type(torch.FloatTensor)
            return text_tokens, mask, visual_prefix, index
        elif self.phase == 'test':
            visual_prefix = self.read_pose_files(index)
            visual_prefix[:, :, :2] = self.normalize_joints(visual_prefix[:, :, :2])
            # reorder [T V C] -> [C T V]
            visual_prefix = np.transpose(visual_prefix, (2, 0, 1))
            visual_prefix = torch.from_numpy(visual_prefix)
            visual_prefix = visual_prefix.type(torch.FloatTensor)
            return visual_prefix, index


model = PoseBackboneWrapper()
model = model.to('cuda:0')
tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir='/mnt/workspace/HF-Models')
args = {
        'feat_path':'/mnt/workspace/How2Sign/openpose_output/json',
        'eos_token':'<|endoftext|>',
}
dataset = Phoenix2014TDataset(args, phoenix_root='', tokenizer=tokenizer, phase='test', split='train')
dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        drop_last=False, 
        sampler=None)

class_path = '/mnt/workspace/CTR-GCN/wlasl2000_label.txt'
with open(class_path, 'r') as f:
    vocab = f.readlines()
    vocab = [x.strip() for x in vocab]
print(len(vocab))
vocab = np.array(vocab)

def gen_slide(length, span=8, step=2):
    if length <= span:
        diff = span - length
        idxs = np.array(range(length))
        idxs = np.concatenate((idxs, (length-1)*np.ones(diff)))
        idxs = idxs[None,:]
    else:
        num_clips = (length - span + (step - 1)) // step + 1
        offsets = np.arange(num_clips)[:,None] * step
        idxs = offsets + np.arange(span)[None, :]
    # idxs = np.mod(idxs, length) # ensure no out of bounds
    idxs = idxs.clip(max=length-1) 
    return idxs

output_pair = []
progress = tqdm(total=len(dataloader))
model.eval()
with torch.no_grad():
    for idx, (prefix, index) in enumerate(dataloader):
        prefix = prefix.to('cuda:0') # B C T V
        T = prefix.shape[2]
        idxs = gen_slide(T, span=24, step=8)
        prefix = prefix[:, :, idxs, :] # B C T' Span V
        B, C, T, S, V = prefix.shape
        prefix = prefix.reshape(C, -1, S, V)
        prefix = prefix.transpose(1, 0)
        output = model(prefix) # B*T' S Class
        output = output.mean(-2) # BT' Class
        idxs = output.argmax(dim=-1).squeeze().cpu().numpy()
        tgt = dataset.translation[index]
        slide_out = vocab[idxs]
        out = []
        for i in range(1, len(slide_out)):
            if len(out) == 0:
                out.append(slide_out[i-1])
            if out[-1] != slide_out[i]:
                out.append(slide_out[i])
        out = ' '.join(out)
        output_pair.append((out, tgt))
        progress.update()

with open('./sweep_input_train_span24_step8_op78_fix.txt', 'w') as f:
    for out, tgt in output_pair:
        f.write(f'{out}|{tgt}\n')