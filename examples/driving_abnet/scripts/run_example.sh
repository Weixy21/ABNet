#!/bin/bash
export PYTHONNOUSERSITE=True  # for python
export PYOPENGL_PLATFORM=egl  # for Vista (cannot connect to "%s"' % name)
args=(
    # trainer
    --default_root_dir $PWD/results
    --exp_name abnet10
    --gpus "1," # use GPU with ID 0
    --max_epochs 10
    --benchmark
    --detect_anomaly
    # logging
    --val_check_interval 0.5
    --log_every_n_steps 50
    --enable_checkpointing
    --enable_model_summary
    # data
    --data-module sequence_data_module
    --data-src $HOME/Python_work/driving/data
    --batch-size 48    # 64
    --num-workers 8    # 8
    --sequence-size 16  # 0-10 (ABNet10) = 32, 10-20 = 16
    # model
    --model-module abnet
    --lr 0.001
    --output-mode "v" "delta" "a" "omega"
    --loss-coef-v 0.25
    --loss-coef-delta 0.25
    --loss-coef-a 0.25
    --loss-coef-omega 0.25
    --cnn-dropout 0.3
    --lstm-size 64
    --q-mlp-dropout 0.3
    --abnet-heads 10
    # --use-attention
)
args+=("$@")

python ../run.py "${args[@]}"
