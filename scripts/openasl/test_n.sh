# 512_pemb_bs48_ep400_encpenc_maskenc_lr3e4_ddp4_dp01_4pt_ccl10m4_e1
python train_openasl_pose_DDP_inter_VN.py \
    --ngpus 1 \
    --work_dir_prefix "/mnt/user/E-linkezhou.lkz-385206/workspace/GloFE/work_dir" \
    --work_dir "openasl/n_model" \
    --tokenizer "notebooks/openasl-v1.0/openasl-bpe25000-tokenizer-uncased" \
    --bs 48 \
    --prefix test-n \
    --phase test --weights "work_dir/openasl/n_model/glofe_n_openasl.pt"

