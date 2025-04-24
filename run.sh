# style
python st_ps.py \
        +data=style \
        +model=stable-diffusion-v1.5-style \
        +task=style_transfer \
        +sampler=sd_edm_daps \
        +reward=style \
        seed=8 \
        task_group=sd \
        save_dir=results/sd/style \
        num_runs=1 \
        name=style_transfer \
        data.start_id=0 \
        data.end_id=9 \
        gpu=0;