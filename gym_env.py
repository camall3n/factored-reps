import warnings
from typing import Tuple, Optional

from cv2 import resize, INTER_AREA, INTER_LINEAR
import gym
from gym import spaces
from gym.core import ObsType, ActType
import matplotlib.pyplot as plt
import numpy as np


from .. import utils


class PendulumEnv(gym.Env):

    _gravity_const = 9.81
    

    def __init__(self, exploring_starts: bool = True, start_position: Tuple = None,
                 terminate_on_goal: bool = True,
                 hidden_goal: bool = False,
                 init_state: Tuple = None,
                 should_render: bool = True):
    
        #TODO: make sure should_render implemented properly (used be initialize_env) so that set_rendering can work

        """
        Pendulum Environment: Gym Wrapper

        Notes:
            removed fixed_goal argument since goal always fixed in pendulum

            dimensions now represents the render size
        """

        self.environment = gym.make('Pendulum-v1', g= self._gravity_const)

        self.exploring_starts = exploring_starts if start_position is None else False
        self.fixed_goal = True
        self.hidden_goal = hidden_goal
        self.terminate_on_goal = terminate_on_goal
        self.should_render = should_render
        

        self.action_space = self.environment.action_space
        self.goal_state = np.array([0.0, 0.0, 0.0])

        self._initialize_state(init_state)
        self._initialize_state_space()
        self._initialize_obs_space()

    
    def _initialize_state_space(self):
        
        #state represented as: (x, y, angular vel) with ranges ([-1,1], [-1,1], [-8,8])
        #TODO: maybe use a Dict representation with tags for each box in state?
        if self.hidden_goal:
            self.factored_state = (spaces.Box(-1,1,shape=(2,)), spaces.Box(-8,8, shape=(1,)))

        if not self.hidden_goal:
            self.factored_state = (spaces.Box(-1,1,shape=(2,)), spaces.Box(-8,8, shape=(1,)), spaces.Box(-1,1,shape=(2,)), spaces.Box(-8,8, shape=(1,)))
        
        '''
        A tuple (more precisely: the cartesian product) of Space instances.

        Elements of this space are tuples of elements of the constituent spaces.    
        '''
        self.state_space = spaces.Tuple(self.factored_state)

    
    def _initialize_obs_space(self):

        env_screen_shape = self.environment.render(mode='rgb_array').shape

        self.img_observation_space = spaces.Box(0.0, 1.0, env_screen_shape, dtype = np.float32)

        #access the already created factored state for making state_space
        #TODO: am I allowed to do this? otherwise need to recreate individual Box sizes to form tuple/dict observation space
        factored_obs_shape = self.factored_state

        self.factor_observation_space = spaces.Tuple(factored_obs_shape)

        self.set_rendering(self.should_render)
        

    def _initialize_env_state(self, init_state = None):

        if init_state is not None:
            self.environment.state = init_state


    def _reset(self):
        pass
    
    def reset(self, seed: Optional[int] = None) -> Tuple[ObsType, dict]:
        self.environment.reset(seed=seed)
        self._cached_state = None
        self._cached_render = None

        self._reset()

        state = self.get_state()
        obs = self.get_observation(state)
        info = self._get_info(state)
        return obs, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        
        self._cached_state = None
        self._cached_render = None

        if self.can_run(action):
            obs, gym_reward, gym_terminated, truncated, info = self._step(action)
        
        state = self.get_state()
        if self.terminate_on_goal and self._check_goal(state):
            terminated = True
        else:
            terminated = False
        
        reward = 1 if terminated else 0
        obs = self.get_observation(state)
        info = self._get_info(state)

        return obs, reward, terminated, truncated, info


    def _step(self, action):
        obs, reward, terminated, truncated, info = self.environment.step(action)
        return obs, reward, terminated, truncated, info
    
    def can_run(self, action):

        return True if (action >= -8.0 and action <= +8.0) else False
    
    def get_state(self):
        
        if self.hidden_goal:
            return self.environment.observation_space
        else:
            return self.environment.observation_space + np.array([0.0,0.0,0.0], dtype = np.float32)
    
    
    def set_state(self, state):
        pass
    
    def get_observation(self, state=None):

        if state is None:
            state = self.get_state()

        
        if self.should_render:
            obs = self._render(state)
        
        else:
            obs = self.get_state()

        return obs
    
    def _get_info(self, state=None):
        if state is None:
            state = self.get_state()
        
        info = {'state': state}
        return info
    
    def _check_goal(self, state=None):
        if state is None:
            state = self.get_state()
        
        for i, val in enumerate(state):

            if val != self.goal_state[i]:
                return False
        
        return True
    
    def set_rendering(self, enabled=True):

        self.should_render = enabled

        if self.should_render:
            self.observation_space = self.img_observation_space
        else:
            self.observation_space = self.factor_observation_space
    
    def plot(self, ob=None, blocking=True):

        if ob is None:
            ob = self.get_observation()
        
        plt.imshow(ob)
        plt.xticks([])
        plt.yticks([])

        if blocking:
            plt.show()
    
    def _render(self, state: Optional[Tuple] = None) -> np.ndarray:
    
    def _resize_if_necessary(self, image, desired_shape):
    
    def _render_composite_image(self, frame: np.ndarray, content: np.ndarray) -> np.ndarray:
    
    def _render_frame(self, content) -> np.ndarray:
    
   