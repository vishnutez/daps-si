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
#         +reward=daps \
#         task_group=pixel \
#         save_dir=results_len_vs_width \
#         num_runs=1 \
#         seed=42 \
#         sampler.annealing_scheduler_config.num_steps=200 \
#         sampler.diffusion_scheduler_config.num_steps=1 \
#         reward.num_particles=8 \
#         data.start_id=0 data.end_id=7 \
#         name=ds_x10_ddim_1_num_8_daps_seed_42 \
#         gpu=0

# # inverse problems
# python mog_si_mult_ps.py \
#         +data=si \
#         +model=ffhq256ddpm \
#         +task=down_sampling \
#         +sampler=edm_daps \
#         +reward=daps \
#         task_group=pixel \
#         save_dir=results_len_vs_width \
#         num_runs=1 \
#         seed=42 \
#         sampler.annealing_scheduler_config.num_steps=200 \
#         sampler.diffusion_scheduler_config.num_steps=1 \
#         task.pixel.mcmc_sampler_config.sigma_thres=0 \
#         reward.num_particles=2 \
#         data.start_id=0 data.end_id=7 \
#         name=ds_x10_ddim_1_num_2_mog_mult_seed_42 \
#         gpu=1

# inverse problems
python mog_si_ps.py \
        +data=si \
        +model=ffhq256ddpm \
        +task=down_sampling \
        +sampler=edm_daps \
        +reward=daps \
        task_group=pixel \
        save_dir=results_len_vs_width \
        num_runs=1 \
        seed=42 \
        sampler.annealing_scheduler_config.num_steps=200 \
        sampler.diffusion_scheduler_config.num_steps=1 \
        task.pixel.mcmc_sampler_config.sigma_thres=0 \
        reward.num_particles=2 \
        data.start_id=0 data.end_id=7 \
        name=ds_x10_ddim_1_num_2_mog_seed_42 \
        gpu=1

