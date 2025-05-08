# inverse problems
python si_ps.py \
        +data=si \
        +model=ffhq256ddpm \
        +task=inpainting_box_deterministic \
        +sampler=edm_daps \
        +reward=daps \
        task_group=pixel \
        save_dir=results_mog_vs_daps \
        num_runs=1 \
        seed=42 \
        sampler.annealing_scheduler_config.num_steps=200 \
        sampler.diffusion_scheduler_config.num_steps=1 \
        reward.num_particles=4 \
        data.start_id=0 data.end_id=7 \
        name=ip_anneal_200_ddim_1_num_4_daps_seed_42 \
        gpu=0