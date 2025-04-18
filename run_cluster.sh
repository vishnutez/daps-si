#!/bin/bash
#SBATCH --job-name=dm      # Job name
#SBATCH --mail-type=BEGIN,END,FAIL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=vishnukunde@tamu.edu  #Where to send mail    
#SBATCH --ntasks=8                      # Run on a 8 cpus (max)
#SBATCH --gres=gpu:a100:1              # Run on a single GPU (max)
#SBATCH --partition=gpu-research                 # Select GPU Partition
#SBATCH --qos=olympus-research-gpu          # Specify GPU queue
#SBATCH --time=2:00:00                 # Time limit hrs:min:sec current 5 min - 36 hour max
#SBATCH --output=logs/%x_%j.out        # Standard output and error log


# select your singularity shell (currently cuda10.2-cudnn7-py36)
singularity shell /mnt/lab_files/ECEN403-404/containers/cuda_10.2-cudnn7-py36.sif

python posterior_sample.py \
        +data=demo-ffhq \
        +model=ffhq256ddpm \
        +task=motion_blur \
        +sampler=edm_daps \
        task_group=ldm \
        save_dir=results \
        num_runs=4 \
        sampler.diffusion_scheduler_config.num_steps=5 \
        sampler.annealing_scheduler_config.num_steps=200 \
        batch_size=2 \
        data.start_id=0 data.end_id=2 \
        name=motion_blur_demo \
        gpu=0;