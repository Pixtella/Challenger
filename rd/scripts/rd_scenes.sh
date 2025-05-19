PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun \
    --standalone --nproc_per_node 4 \
    scripts/adv_inference_magicdrive.py \
    configs/magicdrive/inference/fullx224x400_stdit3_CogVAE_boxTDS_wCT_xCE_wSST.py \
    --cfg-options model.from_pretrained=./ckpts/MagicDriveDiT-stage3-40k-ft/ema.pt \
    num_frames=full cpu_offload=true scheduler.type=rflow-slice --iopth_pth iopths.txt