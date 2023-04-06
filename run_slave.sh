export OMP_NUM_THREADS=20
export NUMEXPR_MAX_THREADS=20
torchrun --nnodes=2 --nproc_per_node=1 --rdzv_id=100 --rdzv_backend=c10d \
  --rdzv_endpoint=10.0.0.8:6035 resnet_ddp.py