# Marin on boa:
source .venv/bin/activate

python experiments/speedrun/hello_world_gpu_speedrun/hello_world_gpu_speedrun.py --prefix local_store


# X & Y axis:
Hardware FLOPs v.s. C4-EN BPB
Note: Hardware FLOPs = Total training time × number of devices × peak FLOPs per device


## The helloworld example with 1gpu and 2gpu shows the difference:

When training the tiny 8.24M parameter model, using 2 GPUs actually performed **worse** than 1 GPU. The 2-GPU run took twice as long (30.5s vs 15.8s) and consumed 4x more Hardware FLOPs (6.04e16 vs 1.56e16), despite doing the same computational work. This happened because Marin uses **data parallelism** - each GPU gets a full copy of the model and processes different batches. For such a small model, the overhead of synchronizing gradients between GPUs, coordinating data loading, and waiting for inter-GPU communication far outweighed any parallelism benefits. Looking at throughput dashboard on wandb, MFU (Model FLOPs Utilization) was much lower with 2 GPUs, indicating the hardware spent more time idle during synchronization than actually computing.

## The hellowword example with 1gpu but 1000 steps instead of 100 steps:
1. Training Hardware FLOPs scales linearly with steps:
100 steps: 1.56e+16 FLOP
1000 steps: 8.13e+16 FLOP (≈5.2x more, close to 10x as expected)

2. Model quality improves (1000 steps gave the model 10x more data to learn from):
C4-EN BPB: 2.45 → 2.033 (lower is better)