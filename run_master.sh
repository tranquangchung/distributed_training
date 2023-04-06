#python -m torch.distributed.launch --nproc_per_node=2 \
#           --nnodes=2 --node_rank=0 --master_addr="10.0.0.8" \
#           --master_port=6050 main.py
#export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet,wl,ww,ppp"
##python -m torch.distributed.launch --nproc_per_node=2 \
##  --nnodes=2 --master_addr="10.0.0.8" \
##  --master_port=6050 resnet_ddp.py
#
#torchrun --nproc_per_node=2 \
#  --nnodes=2 --master_addr="10.0.0.8" --node_rank=0 \
#  --master_port=6050 resnet_ddp.py

#export OMP_NUM_THREADS=20
#export NUMEXPR_MAX_THREADS=20
#export NCCL_DEBUG='INFO'
#export TORCH_DISTRIBUTED_DEBUG='INFO'

#torchrun --nnodes=2 --nproc_per_node=2 --rdzv_id=100 \
#  --rdzv_backend=c10d --rdzv_endpoint=10.0.0.8:29400 main.py

export OMP_NUM_THREADS=20
export NUMEXPR_MAX_THREADS=20
torchrun --nnodes=2 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d \
  --rdzv_endpoint=10.0.0.8:6035 resnet_ddp.py