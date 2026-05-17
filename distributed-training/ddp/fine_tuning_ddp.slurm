#!/bin/bash
#SBATCH --job-name=ddp-llama-fft
#SBATCH --out=%j.out
#SBATCH --err=%j.err
#SBATCH --account=tra26_minwinsc
#SBATCH --reservation=s_tra_minwinsc
#SBATCH --nodes=1                          # change to 2 for the multi-node run
#SBATCH -p boost_usr_prod
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --exclusive

module load cuda
source /leonardo_work/tra26_minwinsc/pyenvs/acceleratenv/bin/activate

export GPUS_PER_NODE=4

# Head node for rendezvous — uses localhost on 1 node, real IP on multi-node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head node IP: $head_node_ip  |  Nodes: $SLURM_NNODES  |  GPUs/node: $GPUS_PER_NODE"

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:29500 \
    FFT_DDP.py 1 1 --batch_size 2 --grad_accum_steps 4
