from datetime import datetime
import importlib
import os

import gym
import numpy as np
import torch
import wandb
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind, ScaledFloatFrame
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize as VecNormalize_

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass


def wrap_gibson(env):

    # return ScaledFloatFrame(env) # nope nope nope, this is done by the cnn
    return env


def make_env(env_id, seed, rank, log_dir, allow_early_resets, custom_gym, navi, enjoy=False):
    def _thunk():
        print("CUSTOM GYM:", custom_gym)
        if custom_gym is not None and custom_gym != "" and "gibson" not in custom_gym:
            module = importlib.import_module(custom_gym, package=None)
            print("imported env '{}'".format((custom_gym)))

        if "gibson" in custom_gym:
            import gibson_transfer

            if "TwoPlayer" in env_id:
                from gibson_transfer.self_play_policies import POLICY_DIR

                if not enjoy:
                    now = datetime.now()  # current date and time

                    subfolder = f"{env_id}-s{seed}-t{now.strftime('%y%m%d_%H%M%S')}"
                    path = os.path.join(POLICY_DIR, subfolder)
                    os.mkdir(path)
                    print("PPO: using Gibson env with output path:", path)
                else:
                    print(
                        "PPO: using Gibson in playback mode, using main " "directory for opponent policies: ",
                        POLICY_DIR,
                    )

        if env_id.startswith("dm"):
            _, domain, task = env_id.split(".")
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)

        if "gibson" in custom_gym and not enjoy and "TwoPlayer" in env_id:
            env.unwrapped.subfolder = subfolder

        #
        if env_id.startswith("Pupper"):
            env = VideoWrapper(env)

        is_atari = hasattr(gym.envs, "atari") and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)

        env.seed(seed + rank)

        obs_shape = env.observation_space.shape

        if str(env.__class__.__name__).find("TimeLimit") >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=allow_early_resets)

        if not navi:
            if is_atari:
                if len(env.observation_space.shape) == 3:
                    env = wrap_deepmind(env)
            elif "Gibson" in env_id:
                env = wrap_gibson(env)
            elif "Splearn" in env_id:
                pass
            elif len(env.observation_space.shape) == 3:
                raise NotImplementedError(
                    "CNN models work only for atari,\n"
                    "please use a custom wrapper for a custom pixel input env.\n"
                    "See wrap_deepmind for an example."
                )

            # If the input has shape (W,H,3), wrap for PyTorch convolutions
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
                env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(
    env_name,
    seed,
    num_processes,
    gamma,
    log_dir,
    device,
    allow_early_resets,
    custom_gym,
    navi=False,
    num_frame_stack=None,
    enjoy=False,
):
    print(f"=== Making {num_processes} parallel envs with {num_frame_stack} stacked frames")
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, custom_gym, navi=navi, enjoy=enjoy)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        print("ENV: ShmemVecEnv")
        envs = ShmemVecEnv(envs, context="fork")
    else:
        print("ENV: DummyVecEnv")
        envs = DummyVecEnvPPO(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            print("ENV: VecNormalize, ret = False")
            envs = VecNormalize(envs, ret=False)
        else:
            print(f"ENV: VecNormalize, gamma = {gamma}")
            envs = VecNormalize(envs, gamma=gamma)

    print(f"ENV: VecPyTorch")
    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        print(f"ENV: VecPyTorchFrameStack, stack: {num_frame_stack}")
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    # elif not navi and not "Gibson" in env_name and len(envs.observation_space.shape) == 3:
    elif not navi and len(envs.observation_space.shape) == 3:
        print("ENV: VecPyTorchFrameStack, stack: 4")
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info["bad_transition"] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, f"Error: Operation, {str(op)}, must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[self.op[0]], obs_shape[self.op[1]], obs_shape[self.op[2]]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()

        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device("cpu")
        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)

        observation_space = gym.spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, : -self.shape_dim0] = torch.clone(self.stacked_obs[:, self.shape_dim0 :])
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0 :] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0 :] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()


class DummyVecEnvPPO(DummyVecEnv):
    def step_wait(self):
        for e in range(self.num_envs):
            action = self.actions[e]
            # if isinstance(self.envs[e].action_space, spaces.Discrete):
            #    action = int(action)

            obs, self.buf_rews[e], self.buf_dones[e], self.buf_infos[e] = self.envs[e].step(action)

            buf_infos = [d.copy() for d in self.buf_infos]

            if self.buf_dones[e]:
                obs = self.envs[e].reset()
            self._save_obs(e, obs)

        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), buf_infos)


class VideoWrapper(gym.Wrapper):
    """ Gathers up the frames from an episode and allows to upload them to Weights & Biases
    Thanks to @cyrilibrahim for this snippet

    """

    def __init__(self, env):
        super(VideoWrapper, self).__init__(env)
        self.episode_images = []

    def reset(self, **kwargs):
        self.episode_images.clear()
        state = self.env.reset()
        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        frame = self.env.render()
        frame = frame[np.newaxis, :, :, :]
        self.episode_images.append(frame)
        return state, reward, done, info

    def send_wandb_video(self):
        wandb.log({"video": wandb.Video(np.concatenate(self.episode_images), fps=10, format="gif")})
