import os
import numpy as np

debug = 0

wandb = 'pupper4'

N_RUNS = 100
SEEDS = list(range(2))

## env
ACTION_SMOOTHING = [1, 2, 3, 4]
RANDOM_ROT = [0, 1, 10, 100]
ACTION_SCALING = [1.] + list(np.arange(0.05, 0.5, 0.05)) 

## model
HIDDEN_SIZES = [64, 128, 256]
N_LAYERS = [2, 3, 4, 5, 6]
FRAME_STACCS = [1,2,4,8]
NUM_PROCESSES = [1] # >1 doesnt work now
# COEFF_REWARD_RUN = [0.1, 0, 1, 10., 100., 1000.]
COEFF_REWARD_RUN = [1., 10., 100., 1000.]
COEFF_REWARD_STABLE = [0.01, 0.1, 0., 1., 10.]
COEFF_REWARD_CTRL = [0.01, 0.1, 0., 1., 10.]


if debug:
    wandb = 'temp'

for seed in SEEDS:
  for hidden_size in HIDDEN_SIZES:
    for frame_stacc in FRAME_STACCS:
      for num_processes in NUM_PROCESSES:
        for scale_down in SCALE_DOWN:
            command = ("borgy submit -i images.borgy.elementai.net/fgolemo/gym:v4 " 
                "--mem 32 --gpu-mem 12 --gpu 1 --cuda-version 10.1 -H -- bash -c "
                " 'cd /root && export PATH=/mnt/home/optimass/miniconda3/bin/:$PATH " 
                "&& export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/home/optimass/.mujoco/mujoco200/bin " 
                "&& cd ~/StanfordQuadruped/ && pip install -e . " 
                # "&& cd ~/baselines/ && pip install -e . " 
                "&& pip install wandb moviepy imageio "
                "&& cd ~/pytorch-a2c-ppo-acktr-gail && ls . "
                "&& python main.py "
                "--custom-gym stanford_quad "
                ## "Pupper-Walk-{action}_aScale_{action_scaling}-aSmooth_{action_smoothing}-RandomZRot_{random_rot}-{headlessness}-v0
                f"--env-name Pupper-Walk-Relative_aScale_{action_scaling:.2}-aSmooth_{action_smoothing}-RandomZRot_{random_rot}-Headless-v0 "
                f"--action_scaling {action_scaling:.2} "
                f"--action_smoothing {action_smoothing} "
                f"--random_rot {random_rot} "
                "--algo ppo "
                "--use-gae "
                "--log-interval 1 "
                "--num-steps 2048 "
                f"--num-processes {num_processes} "
                "--lr 3e-4 "
                "--entropy-coef 0 "
                "--value-loss-coef 0.5 "
                "--ppo-epoch 10 "
                "--num-mini-batch 32 "
                "--gamma 0.99 "
                "--gae-lambda 0.95 "
                f"--frame-stacc {frame_stacc} "
                #"--num-env-steps 5000000 "
                "--num-env-steps 20000000 "
                "--use-linear-lr-decay "
                "--use-proper-time-limits "
                "--save-interval 10 "
                f"--hidden_size {hidden_size} "
                f"--n_layers {n_layers} "
                f"--seed {seed} "
                f"--wandb {wandb} "
                f"--wandb_name  {wandb_name} "
                "--num-steps 1200 "
                "--gif-interval 100 "
                f"--coeff_reward_run {coeff_reward_run:.2} "
                f"--coeff_reward_stable {coeff_reward_stable:.2} "
                f"--coeff_reward_ctrl {coeff_reward_ctrl:.2} "
                "> ~/borgy/pupper.log '"
        )
          if debug:
              exit()
