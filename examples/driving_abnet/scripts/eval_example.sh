#!/bin/bash

#bash ./scripts/eval_example.sh --trace-paths /home/tsunw/data/vista/traces/20220113-131922_lexus_devens_outerloop_reverse
export PYTHONNOUSERSITE=True   #for python
export PYOPENGL_PLATFORM=osmesa   #for Vista (cannot connect to "%s"' % name)  egl  pyrender 0.1.45 requires PyOpenGL==3.1.0, but you have pyopengl 3.1.4 which is incompatible.

DATA_ROOT=/home/tsunw/data/vista
TRACE_ROOT=$DATA_ROOT/traces

args=(
    # env
    --trace-paths $TRACE_ROOT/20210527-131252_lexus_devens_center_outerloop
                  $TRACE_ROOT/20210527-131709_lexus_devens_center_outerloop_reverse
                  $TRACE_ROOT/20210609-122400_lexus_devens_outerloop_reverse
                  $TRACE_ROOT/20210609-123703_lexus_devens_outerloop
                  $TRACE_ROOT/20210609-133320_lexus_devens_outerloop
                  $TRACE_ROOT/20210609-154745_lexus_devens_outerloop_reverse
                  $TRACE_ROOT/20210609-155238_lexus_devens_outerloop
                  $TRACE_ROOT/20210609-175037_lexus_devens_outerloop_reverse
                  $TRACE_ROOT/20210609-175503_lexus_devens_outerloop
                  $TRACE_ROOT/20210613-171636_lexus_devens_outerloop
                  $TRACE_ROOT/20210613-172102_lexus_devens_outerloop_reverse
                  $TRACE_ROOT/20210613-193528_lexus_devens_outerloop_reverse
                  $TRACE_ROOT/20210726-131322_lexus_devens_center
                  $TRACE_ROOT/20210726-131912_lexus_devens_center_reverse
                  $TRACE_ROOT/20210726-154641_lexus_devens_center
                  $TRACE_ROOT/20210726-155941_lexus_devens_center_reverse
                  $TRACE_ROOT/20210726-184624_lexus_devens_center
                  $TRACE_ROOT/20210726-184956_lexus_devens_center_reverse
                  $TRACE_ROOT/20211114-145502_lexus_devens_outerloop_reverse
                  $TRACE_ROOT/20220113-130450_lexus_devens_outerloop_outdoor_wb
                  $TRACE_ROOT/20220113-130914_lexus_devens_outerloop_reverse_outdoor_wb
                  $TRACE_ROOT/20220113-131457_lexus_devens_outerloop
                  $TRACE_ROOT/20220113-131922_lexus_devens_outerloop_reverse
                  $TRACE_ROOT/20220113-134438_lexus_devens_outerloop_reverse
                  $TRACE_ROOT/20220113-135302_lexus_devens_outerloop
    --mesh-dir $DATA_ROOT/carpack01/
    --n-agents 2
    --reset-mode segment_start # uniform # default
    # --use-curvilinear-dynamics
    --n-episodes 1
    --max-step 500
    --init-dist-range 15.0 25.0 #20.0 20.0 #15.0 25.0 #15 25
    --init-lat-noise-range 1.5 1.5 #1.0 1.0 #0.1 1.5 #1 1.5
    # --use-reference-control

    # logging
    --out-dir ./tmp/test
    --use-display
    --save-video

    # barrier net
    --model-module abnet #abnet  barrier_net  bnet_up
    # --model-module barrier_net 
    # --ckpt scripts/results/abnet10/version_6/checkpoints/epoch\=3-step\=15591.ckpt



    # --ckpt-sc scripts/results/abnet10_att/version_0/checkpoints/epoch\=4-step\=19489.ckpt  # scale

    # --ckpt-sc3 scripts/results/abnet10_att/version_0/checkpoints/epoch\=4-step\=17540.ckpt  # scale

    # --ckpt-sc4 scripts/results/abnet10/version_6/checkpoints/epoch\=4-step\=19489.ckpt  # scale
    # --ckpt-sc5 scripts/results/abnet10_att/version_0/checkpoints/epoch\=3-step\=15591.ckpt  # scale

    # --ckpt-sc6 scripts/results/abnet10/version_6/checkpoints/epoch\=4-step\=17540.ckpt  # scale
    # --ckpt-sc7 scripts/results/abnet10_att/version_0/checkpoints/epoch\=3-step\=13642.ckpt  # scale

    # --ckpt-sc8 scripts/results/abnet10/version_6/checkpoints/epoch\=3-step\=13642.ckpt  # scale
    # --ckpt-sc9 scripts/results/abnet10_att/version_0/checkpoints/epoch\=2-step\=11693.ckpt  # scale
    # --ckpt-sc10 scripts/results/abnet10/version_5/checkpoints/epoch\=4-step\=19489.ckpt  # scale

    --ckpt scripts/results/abnet10_att/version_0/checkpoints/epoch\=4-step\=19489.ckpt
    # --ckpt scripts/results/abnet10_att/version_2/checkpoints/epoch\=7-step\=29234.ckpt
    # --ckpt scripts/results/epoch\=9-step\=29239.ckpt
    # --ckpt scripts/results/bnet/version_5/checkpoints/epoch\=9-step\=38979.ckpt
    --set-obs-d-lower-bound 1

    #6 abnet-att, 7 abnet, 8 bnet, 9 dfb, 10 e2e
    # state net
    # --state-net-model-module state_net
    # --state-net-ckpt /home/tsunw/results/bnet/state_net_9/version_1/checkpoints/epoch=7-step=36374.ckpt
)
args+=("$@")

python eval.py "${args[@]}"     #NOTE here 