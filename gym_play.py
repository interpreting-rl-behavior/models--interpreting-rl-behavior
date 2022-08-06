import numpy as np
import gym
from gym3.interop import ToBaselinesVecEnv
from common.env.procgen_wrappers import *
import matplotlib.pyplot as plt

from custom_envs import MountainCarPixelEnv

"""
http://incompleteideas.net/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
import math
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.error import DependencyNotInstalled


class MountainCarEnvReward(gym.Env):
    """
    ### Description

    The Mountain Car MDP is a deterministic MDP that consists of a car placed stochastically
    at the bottom of a sinusoidal valley, with the only possible actions being the accelerations
    that can be applied to the car in either direction. The goal of the MDP is to strategically
    accelerate the car to reach the goal state on top of the right hill. There are two versions
    of the mountain car domain in gym: one with discrete actions and one with continuous.
    This version is the one with discrete actions.

    This MDP first appeared in [Andrew Moore's PhD Thesis (1990)](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-209.pdf)

    ```
    @TECHREPORT{Moore90efficientmemory-based,
        author = {Andrew William Moore},
        title = {Efficient Memory-based Learning for Robot Control},
        institution = {University of Cambridge},
        year = {1990}
    }
    ```

    ### Observation Space

    The observation is a `ndarray` with shape `(2,)` where the elements correspond to the following:

    | Num | Observation                          | Min  | Max | Unit         |
    |-----|--------------------------------------|------|-----|--------------|
    | 0   | position of the car along the x-axis | -Inf | Inf | position (m) |
    | 1   | velocity of the car                  | -Inf | Inf | position (m) |

    ### Action Space

    There are 3 discrete deterministic actions:

    | Num | Observation             | Value | Unit         |
    |-----|-------------------------|------ |--------------|
    | 0   | Accelerate to the left  | Inf   | position (m) |
    | 1   | Don't accelerate        | Inf   | position (m) |
    | 2   | Accelerate to the right | Inf   | position (m) |

    ### Transition Dynamics:

    Given an action, the mountain car follows the following transition dynamics:

    *velocity<sub>t+1</sub> = velocity<sub>t</sub> + (action - 1) * force - cos(3 * position<sub>t</sub>) * gravity*

    *position<sub>t+1</sub> = position<sub>t</sub> + velocity<sub>t+1</sub>*

    where force = 0.001 and gravity = 0.0025. The collisions at either end are inelastic with the velocity set to 0
    upon collision with the wall. The position is clipped to the range `[-1.2, 0.6]` and
    velocity is clipped to the range `[-0.07, 0.07]`.


    ### Reward:

    The goal is to reach the flag placed on top of the right hill as quickly as possible, as such the agent is
    penalised with a reward of -1 for each timestep it isn't at the goal and is not penalised (reward = 0) for
    when it reaches the goal.

    ### Starting State

    The position of the car is assigned a uniform random value in *[-0.6 , -0.4]*.
    The starting velocity of the car is always assigned to 0.

    ### Episode Termination

    The episode terminates if either of the following happens:
    1. The position of the car is greater than or equal to 0.5 (the goal position on top of the right hill)
    2. The length of the episode is 200.


    ### Arguments

    ```
    gym.make('MountainCar-v0')
    ```

    ### Version History

    * v0: Initial versions release (1.0.0)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, goal_velocity=0):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity

        self.force = 0.001
        self.gravity = 0.0025

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        self.screen = None
        self.clock = None
        self.isopen = True

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def step(self, action: int):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        # reward = -1.0
        reward = 10. if done else 0.

        self.state = (position, velocity)
        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}

    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55

    def render(self, mode="human"):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 40
        carheight = 20
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        pos = self.state[0]

        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, ys * scale))

        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        clearance = 10

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            coords.append(
                (
                    c[0] + (pos - self.min_position) * scale,
                    c[1] + clearance + self._height(pos) * scale,
                )
            )

        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))

        for c in [(carwidth / 4, 0), (-carwidth / 4, 0)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos))
            wheel = (
                int(c[0] + (pos - self.min_position) * scale),
                int(c[1] + clearance + self._height(pos) * scale),
            )

            gfxdraw.aacircle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )
            gfxdraw.filled_circle(
                self.surf, wheel[0], wheel[1], int(carheight / 2.5), (128, 128, 128)
            )

        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def get_keys_to_action(self):
        # Control with left and right arrow keys.
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


pub = lambda x: [i for i in dir(x) if not i.startswith("_")]
# Must "make" the mountaincar in order for its module to be accessible
# gym.make("MountainCar-v0")

# class MountainCarPixel(gym.envs.classic_control.mountain_car.MountainCarEnv):
#     def __init__(self):
#         super().__init__()

#     def step(self, action):
#         _, reward, done, _ = super().step(action)
#         obs = self.render(mode="rgb_array")
#         return obs, reward, done, {}


class ResizeObservationVec(gym.wrappers.ResizeObservation):
    def observation(self, observation):
        """Updates the observations by resizing the observation to shape given by :attr:`shape`.

        Args:
            observation: The observation to reshape

        Returns:
            The reshaped observations

        Raises:
            DependencyNotInstalled: opencv-python is not installed
        """
        try:
            import cv2
        except ImportError:
            raise gym.error.DependencyNotInstalled(
                "opencv is not install, run `pip install gym[other]`"
            )

        resized_imgs = []
        for img_idx in range(observation.shape[0]):
            resized_imgs.append(cv2.resize(
                observation[img_idx], self.shape[::-1], interpolation=cv2.INTER_AREA
            ))
        observation = np.stack(resized_imgs, axis=0)

        # observation = cv2.resize(
        #     observation, self.shape[::-1], interpolation=cv2.INTER_AREA
        # )

        # if observation.ndim == 2:
        if observation.ndim == 3:
            observation = np.expand_dims(observation, -1)
        return observation

    def step_wait(self):
        obs, reward, done, info = self.env.step_wait()
        return self.observation(obs), reward, done, info

# class TransformReward(gym.wrappers.RewardWrapper):
#     def reward(self, reward):

def sample_n(space, n=2):
    samples = []
    for i in range(n):
        samples.append(space.sample())
    return np.array(samples)

if __name__ == "__main__":

    max_steps = 200
    gym.envs.register(
        id='MountainCarLongHorizon-v0',
        # entry_point='gym.envs.classic_control:MountainCarEnv',
        entry_point='gym_play:MountainCarEnvReward',
        max_episode_steps=max_steps,      # MountainCar-v0 uses 200
        reward_threshold=-110.0,
    )
    # We should change the reward to 10 (like coinrun)
    env = gym.make('MountainCarLongHorizon-v0')

    # num_envs = 2
    # # env = create_venv_gym({}, {})
    # env = gym.make("MountainCar-v0")
    # print(env.spec.max_episode_steps)
    # env.spec.max_episode_steps = 50
    # print(env.spec.max_episode_steps)
    # env._max_episode_length = 30
    # import gym3
    # env_fn = lambda: MountainCarPixelEnv()
    # # env = gym3.vectorize_gym(num=2, render_mode="human", env_kwargs={"id": "CartPole-v0"})
    # venv = gym3.vectorize_gym(num=num_envs, env_fn=env_fn, render_mode="rgb_array")

    # # env = MountainCarPixelEnv()
    # venv = ToBaselinesVecEnv(venv)
    # # venv = gym.wrappers.ResizeObservation(venv, (64, 64))
    # venv = ResizeObservationVec(venv, (64, 64))
    # venv = TransposeFrame(venv) # observation_space: Box(0., 255., (3, 64, 64), float32)
    # venv = ScaledFloatFrame(venv) # observation_space: Box(0., 1., (3, 64, 64), float32)
    # observation, info = env.reset(seed=42, return_info=True)
    # This is only return one observation, not num=2.
    # observation = venv.reset()
    observation = env.reset()
    dones = 0
    num_steps = 1000000
    # num_steps = 300
    episode_steps = 0
    for i in range(num_steps):
        # venv.render()
        # env.render()
        # Multiple actions
        # action = sample_n(venv.action_space, num_envs)
        action = env.action_space.sample()
        # policy(observation)  # User-defined policy function
        observation, reward, done, info = env.step(action)
        # if reward != -1:
            # print("HERER")
        # if i % 20 == 0:
            # print(reward)
            # for env_id in range(num_envs):
            #     print(observation.shape)
            #     print("Env: ", env_id)
            #     plt.imshow(observation[env_id].transpose(1,2,0))
            #     plt.show()
        # if done.any():
        # print(i)
        episode_steps += 1
        if done:
            # if (i + 1) % max_steps > 0:
            if episode_steps % max_steps > 0:
                print("COMPLETED ONE")
                dones += 1
            observation, info = env.reset(return_info=True)
            episode_steps = 0
    env.close()
    print(f"completed {dones}/{num_steps // max_steps} episodes")

# OrderedDict([('rgb', Box(0, 255, (64, 64,...3), uint8))])
# venv.observation_space["rgb"]
# Box(0, 255, (64, 64, 3), uint8)
# venv.observation_space.spaces["rgb"]