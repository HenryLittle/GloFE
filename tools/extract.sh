MMPOSE_ROOT=/mnt/workspace
python extract_phoenix2014T_mp.py \
    $MMPOSE_ROOT/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    $MMPOSE_ROOT/mmpose/models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    $MMPOSE_ROOT/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark.py \
    $MMPOSE_ROOT/mmpose/models/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
    --sid $1 \
    --splits $2 \
    --device cuda:$3


# Model urls
# https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth
