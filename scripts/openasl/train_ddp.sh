GPUS=2
torchrun --nnodes=1 --nproc_per_node=$GPUS train_openasl_pose_DDP_inter_VN.py \
    --ngpus $GPUS \
    --work_dir_prefix "/mnt/user/E-linkezhou.lkz-385206/workspace/GloFE/work_dir" \
    --work_dir "openasl/train_test" \
    --bs 48 --ls 0.2 --epochs 400 \
    --save_every 5 \
    --clip_length 512 --vocab_size 25000 \
    --feat_path "/mnt/user/E-linkezhou.lkz-385206/workspace/OpenASL/mmpose" \
    --label_path "/mnt/user/E-linkezhou.lkz-385206/workspace/openasl-pre/openasl-v1.0.tsv" \
    --eos_token "</s>" \
    --tokenizer "notebooks/openasl-v1.0/openasl-bpe25000-tokenizer-uncased" \
    --pose_backbone "PartedPoseBackbone" \
    --pe_enc --mask_enc --lr 3e-4 --dropout_dec 0.1 --dropout_enc 0.1 \
    --inter_cl --inter_cl_margin 0.4 --inter_cl_alpha 1.0 \
    --inter_cl_vocab 5523 \
    --inter_cl_we_path "notebooks/openasl-v1.0/uncased_filtred_glove_VN_embed.pkl"