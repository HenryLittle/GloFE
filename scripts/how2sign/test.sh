# 512_pemb_bs40_ep200_encpenc_maskenc_lr3e4_ddp8_dp01_4pt_icl10m5_e1
python train_how2_pose_DDP_inter_VN.py \
    --ngpus 1 \
    --work_dir_prefix "/mnt/user/E-linkezhou.lkz-385206/workspace/GloFE/work_dir" \
    --work_dir "how2sign/vn_model" \
    --tokenizer "notebooks/how2sign/how2sign-bpe25000-tokenizer-uncased" \
    --bs 40 \
    --prefix test-vn\
    --phase test --weights "work_dir/how2sign/vn_model/glofe_vn_how2sign_0224.pt"

