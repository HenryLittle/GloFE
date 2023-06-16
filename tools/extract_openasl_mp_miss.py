# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser

import cv2
import mmcv

from mmpose.apis import (collect_multi_frames, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import numpy as np
import pickle as pkl

def main():
    """Visualize the demo video (support both single-frame and multi-frame).

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.2,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    parser.add_argument(
        '--use-multi-frames',
        action='store_true',
        default=False,
        help='whether to use multi frames for inference in the pose'
        'estimation stage. Default: False.')
    parser.add_argument(
        '--online',
        action='store_true',
        default=False,
        help='inference mode. If set to True, can not use future frame'
        'information when using multi frames for inference in the pose'
        'estimation stage. Default: False.')
    parser.add_argument(
        '--sid',
        type=int,
        default=0)
    parser.add_argument(
        '--splits',
        type=int,
        default=1)

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    # assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    print('Initializing model...')
    # build the detection model from a config file and a checkpoint file
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())

    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    # get datasetinfo
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    arg_dict = {
        'det_model': det_model,
        'pose_model': pose_model,
        'dataset': dataset,
        'dataset_info': dataset_info,
        'output_root': '/mnt/workspace/OpenASL/mmpose',
        'args': args,
    }

    all_samples = load_sample_names('/mnt/workspace/openasl-pre/notebooks/missing_sample_clip-v1.0.txt')
    # all_samples = all_samples[:5]
    total_samples = len(all_samples)
    print('Total samples:', total_samples)
    chunk = (total_samples + args.splits - 1) // args.splits
    sample_split = all_samples[args.sid * chunk: min((args.sid + 1) * chunk, total_samples)]
    print(f'Running split:[{args.sid * chunk}:{min((args.sid + 1) * chunk, total_samples)}]')

    for sample_vid in tqdm(sample_split):
        sample_id = sample_vid.split('/')[-1][:-4]
        output_file_path = os.path.join(arg_dict['output_root'], f'{sample_id}.pkl')
        if os.path.exists(output_file_path):
            continue
        process_single_video(sample_vid, arg_dict)
    


def process_single_video(video_path, arg_dict):
    det_model = arg_dict['det_model']
    pose_model = arg_dict['pose_model']
    dataset = arg_dict['dataset']
    dataset_info = arg_dict['dataset_info']
    output_root = arg_dict['output_root']
    args = arg_dict['args']
    # read video
    video = mmcv.VideoReader(video_path)
    # assert video.opened, f'Faild to load video file {video_path}'
    if not video.opened:
        with open(f'log-open-failed-s{args.sid}.txt', 'a') as f:
            f.write(f'Failed to load video file: {video_path}\n')
        return

    # if args.out_video_root == '':
    #     save_out_video = False
    # else:
    #     os.makedirs(args.out_video_root, exist_ok=True)
    #     save_out_video = True

    # if save_out_video:
    #     fps = video.fps
    #     size = (video.width, video.height)
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     videoWriter = cv2.VideoWriter(
    #         os.path.join(args.out_video_root,
    #                      f'vis_{os.path.basename(args.video_path)}'), fourcc,
    #         fps, size)

    # frame index offsets for inference, used in multi-frame inference setting
    # if args.use_multi_frames:
    #     assert 'frame_indices_test' in pose_model.cfg.data.test.data_cfg
    #     indices = pose_model.cfg.data.test.data_cfg['frame_indices_test']

    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None
    results = []
    sample_id = video_path.split('/')[-1][:-4]
    # print('Running inference...')
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        # get the detection results of current frame
        # the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, cur_frame)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

        if args.use_multi_frames:
            frames = collect_multi_frames(video, frame_id, indices,
                                          args.online)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            frames if args.use_multi_frames else cur_frame,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        if len(pose_results) != 0:
            pose_results = pose_results[0]
            results.append(pose_results['keypoints'])
        else:
            print(f'{sample_id} Frame:{frame_id} has no person')
            with open(f'log-ext-openasl-s{args.sid}.txt', 'a') as f:
                f.write(f'{sample_id} Frame:{frame_id} has no person\n')

    results = np.array(results)
    with open(os.path.join(output_root, f'{sample_id}.pkl'), 'wb') as f:
        if len(results.shape) == 3:
            pkl.dump(results, f)
        else:
            with open(f'log-ext-openasl-pvid-s{args.sid}.txt', 'a') as f:
                f.write(f'{sample_id} Incorrect result shape: {results.shape}\n')



def list_all_vid_names(root_dir, out_list_file):
    paths = []
    samples = sorted(os.listdir(root_dir))
    for sample in samples:
        paths.append(os.path.join(root_dir, sample))
    with open(out_list_file, 'w') as f:
        f.writelines('\n'.join(paths))

def load_sample_names(txt_path):
    with open(txt_path, 'r') as f:
        paths = f.readlines()
    paths = [x.strip() for x in paths]
    print('Total samples:', len(paths))
    return paths


if __name__ == '__main__':
    main()
    # list_all_vid_names('/mnt/workspace/OpenASL/video-clips', 'open_asl_samples.txt')
