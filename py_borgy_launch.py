import os
import numpy as np
import string
import random

def id_generator(size=8, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

debug = 0

wandb = 'pupper4'

N_RUNS = 250
SEEDS = list(range(2))

## env
# ACTION_TYPE = ["Relative", "Incremental"]

# ACTION_TYPE = ["Relative"]
# ACTION_SCALING = [1.0, 2.0, 4.0] + list(np.arange(0.05, 0.5, 0.05))
# ACTION_SMOOTHING = [1, 2, 3, 4]

ACTION_TYPE = ["Incremental"]
ACTION_SCALING = [1.0, 2.0, 0.45]
ACTION_SMOOTHING = [1,]

RANDOM_ROT = [0, 1, 10, 100]
# episode_length = [120, 240, 360]
EPISODE_LENGTH = [240, 360]

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

for _ in range(N_RUNS):

    seed = np.random.choice(SEEDS)
    
    action_type = np.random.choice(ACTION_TYPE)
    action_scaling = np.random.choice(ACTION_SCALING)
    action_smoothing = np.random.choice(ACTION_SMOOTHING)
    random_rot = np.random.choice(RANDOM_ROT)
    episode_length = np.random.choice(EPISODE_LENGTH)
    
    hidden_size = np.random.choice(HIDDEN_SIZES)
    frame_stacc = np.random.choice(FRAME_STACCS)
    num_processes = np.random.choice(NUM_PROCESSES)
    n_layers = np.random.choice(N_LAYERS)
    coeff_reward_run = np.random.choice(COEFF_REWARD_RUN)
    coeff_reward_stable = np.random.choice(COEFF_REWARD_STABLE)
    coeff_reward_ctrl = np.random.choice(COEFF_REWARD_CTRL)

    wandb_name = id_generator() 

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
        ## "Pupper-Walk-Relative-aScale_0.05-aSmooth_4-RandomZRot_1-Headless-v0"
        # name = (
        #     f"Pupper-Walk-{action_type}-"
        #     f"steps_{episode_length}-"
        #     f"aScale_{action_scaling:.2}-"
        #     f"aSmooth_{action_smoothing}-"
        #     f"RandomZRot_{random_rot}-{headlessness}-v0"
        # )
        f"--env-name Pupper-Walk-{action_type}-steps_{episode_length}-aScale_{action_scaling:.2}-aSmooth_{action_smoothing}-RandomZRot_{random_rot}-Headless-v0 "
        f"--action_scaling {action_scaling:.2} "
        f"--action_smoothing {action_smoothing} "
        f"--random_rot {random_rot} "
        f"--action_type {action_type} "
        f"--episode_length {episode_length} "
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
    
    print(command)
    os.system(command)

    if debug:
        os._exit(0)

