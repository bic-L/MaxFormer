export HOST_NODE_ADDR=127.0.0.1:2963
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_SOCKET_IFNAME=lo

name=train_10_512

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=$HOST_NODE_ADDR test.py\
 --pin_mem --dist_eval -c ./conf/10_512_t1.yml --exp $name --log_dir ./log/$name --output_dir ./output/$name --resume ./checkpoints/10-512-T1.pth.tar
