# # style_transfer
# python st_ps.py \
#         +data=style \
#         +model=stable-diffusion-v1.5-style \
#         +task=style_transfer \
#         +sampler=sd_edm_daps \
#         +reward=style \
#         seed=8 \
#         task_group=sd \
#         save_dir=results/sd/style \
#         num_runs=1 \
#         name=style_transfer \
#         data.start_id=2 \
#         data.end_id=3 \
#         gpu=0;

# # inverse problems
# python si_ps.py \
#         +data=si \
#         +model=ffhq256ddpm \
#         +task=down_sampling \
#         +sampler=edm_daps \
#         +reward=daps_resample \
#         task_group=pixel \
#         save_dir=results_len_vs_width \
#         num_runs=1 \
#         seed=42 \
#         sampler.annealing_scheduler_config.num_steps=200 \
#         sampler.diffusion_scheduler_config.num_steps=1 \
#         reward.num_particles=4 \
#         data.start_id=0 data.end_id=7 \
#         name=ds_x10_ddim_1_num_4_daps_seed_42 \
#         gpu=0

# # inverse problems
# python si_ps.py \
#         +data=si \
#         +model=ffhq256ddpm \
#         +task=down_sampling \
#         +sampler=edm_daps \
#         +reward=daps_resample \
#         task_group=pixel \
#         save_dir=results_len_vs_width \
#         num_runs=1 \
#         seed=42 \
#         sampler.annealing_scheduler_config.num_steps=200 \
#         sampler.diffusion_scheduler_config.num_steps=1 \
#         reward.num_particles=4 \
#         data.start_id=0 data.end_id=7 \
#         name=ds_x10_ddim_1_num_4_resample_seed_42_recheck_2 \
#         gpu=1

# inverse problems
python mog_si_ps.py \
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
        task.pixel.mcmc_sampler_config.mult_init=False \
        reward.num_particles=4 \
        data.start_id=0 data.end_id=7 \
        name=ip_anneal_200_ddim_1_num_4_mog_single_init_seed_42 \
        gpu=1

