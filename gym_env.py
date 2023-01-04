import warnings
from typing import Tuple, Optional

from cv2 import resize, INTER_AREA, INTER_LINEAR
import gym
from gym import spaces
from gym.core import ObsType, ActType
import matplotlib.pyplot as plt
import numpy as np
import copy
import pdb


# from .. import utils


class PendulumEnv(gym.Env):

    _gravity_const = 9.81
    

    def __init__(self, exploring_starts: bool = True, start_position: Tuple = None,
                 terminate_on_goal: bool = True,
                 init_state: Tuple = None,
                 should_render: bool = True):
    

        """
        Pendulum Environment: Gym Wrapper

        Notes:
            removed fixed_goal argument since goal always fixed in pendulum

            dimensions now represents the render size
        """

        #TODO: removed teh gym.make().unwrapped
        self.environment = gym.make('Pendulum-v1', g= self._gravity_const)
        self._initial_state = None

        self.exploring_starts = exploring_starts if start_position is None else False
        self.fixed_goal = True
        #self.hidden_goal = hidden_goal
        self.terminate_on_goal = terminate_on_goal
        self.should_render = should_render
        

        self.action_space = self.environment.action_space
        self.goal_state = np.array([0.0, 0.0, 0.0])
        self.current_state = None

        self._initialize_env_state(init_state)
        self._initialize_state_space()
        self._initialize_obs_space()

    
    def _initialize_state_space(self):
        
        #state represented as: (x, y, angular vel) with ranges ([-1,1], [-1,1], [-8,8])
        #TODO: self.environment.state missing the angular velocity so left as the x,y coord
        self.state_space = spaces.Box(low=np.array([-1,-1,-8]), high=np.array([1,1,8]))

    
    def _initialize_obs_space(self):
        
        #TODO: Do we need mode = 'rgb_array' because it works without?
        env_screen_shape = self.environment.render().shape

        self.img_observation_space = spaces.Box(0.0, 1.0, env_screen_shape, dtype = np.float32)

        #access the already created factored state for making state_space
        self.factor_observation_space = copy.deepcopy(self.state_space)

        self.set_rendering(self.should_render)
        

    def _initialize_env_state(self, init_state = None):

        if init_state is not None:
            self.environment.state = init_state
            self._initial_state = init_state
        else:
            self.environment.reset()

    
    def _reset(self, init_state):
        self._initial_state = init_state
        self.current_state = init_state
    
    def reset(self, seed: Optional[int] = None) -> Tuple[ObsType, dict]:

        super().reset(seed=seed)
        new_obs, _ = self.environment.reset(seed=seed)
        self._cached_state = None
        self._cached_render = None

        #TODO: I use the private _reset() function to update the self._initial_state reference
        # state = self.get_state()
        self._reset(new_obs)
        
        obs = self.get_observation(new_obs)
        info = self._get_info(new_obs)
        return obs, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        
        self._cached_state = None
        self._cached_render = None

        if self.can_run(action):
            state, gym_reward, gym_terminated, truncated, info = self._step(action)
        
        self.current_state = state
        
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

        return True if (action >= -2.0 and action <= +2.0) else False
    
    def get_state(self):

        # theta, angular_vel = self.environment.state[0], self.environment.state[1]
        
        # return np.asarray([np.cos(theta), np.sin(theta), angular_vel])

        return self.current_state
       
    
    
    def set_state(self, state):

        #NOTE: state property in self.environment is tuple (theta, angular velocity)
        
        is_valid = self._check_valid_state(state)

        assert is_valid, 'Attempted to call set_state with an invalid state'

        #convert state to theta, angular vel form for self.environment internal state
        x, y, angular_vel = state[0], state[1], state[2]

        theta = np.arctan2(y,x)

        self.environment.state = np.array([theta, angular_vel])
        self._initial_state = state
        self.current_state = state
    
    def _check_valid_state(self, state):
        
        is_valid = self.state_space.contains(list(state)) if state is not None else False

        return is_valid
    
    def is_valid_state(self, state):
        return self._check_valid_state(state)
    
    def get_observation(self, state=None):

        if state is None:
            state = self.get_state()

        if self.should_render:
            obs = self._render(state)
        else:
            obs = state

        return obs
    
    def _render(self, state=None):
        
        current_state = self.get_state()

        try:
            if state is not None:
                self.set_state(state)

            if (self._cached_state is None) or (state != self._cached_state).any():
                self._cached_state = state
                self._cached_render = self.environment.render()
            
            return self._cached_render

        finally:
            self.set_state(current_state)
        
    
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
    
    def plot(self, ob=None, blocking=True, save=False, filename=None):

        assert self.should_render is True

        if ob is None:
            ob = self.get_observation()
        
        plt.imshow(ob)
        plt.xticks([])
        plt.yticks([])

        if blocking:
            plt.show()
        
        if save:

            if filename is None:
                raise Exception('Error in plot(): no filename provided but save = True')
            plt.savefig(filename)
        
        plt.cla()
        plt.clf()



def test_file():

    #creating the environment
    test_env = PendulumEnv()

    

    print('State Space: ', test_env.state_space)
    print('Action Space: ', test_env.action_space)
    print('-------------------------------------')
    print('Factored Obs Space: ', test_env.factor_observation_space)
    print('Image Obs Space: ', test_env.img_observation_space)
    print('Observation Space: {} | Should Render: {}'.format(test_env.observation_space, test_env.should_render))

    #resetting the environment
    try:
        test_env.reset()
    except Exception as e:
        print('Error: {}'.format(e))
    
    
    assert test_env.can_run(1.8234) is True
    assert test_env.can_run(-10.01) is False
    assert test_env.can_run(2.0000000001) is False

    
    curr_state = test_env.get_state()
    print('Current State (get_state): ', curr_state)

    current_obs = test_env.get_observation()
    print('Current Observation: type: {} | shape: {} | values: {}'.format(type(current_obs), current_obs.shape, current_obs))

    new_action = test_env.action_space.sample()
    print('Sampled Action: {}'.format(new_action))
    
    test_env.plot(blocking=False, save=True, filename='./test/old_obs.png')

    
    obs, reward, terminated, truncated, info = test_env.step(new_action)
    print('New Post-Action State (get_state): ', test_env.get_state())

    test_env.plot(blocking=False, save=True, filename='./test/action_applied.png')

    for i in range(10):
        obs, reward, terminated, truncated, info = test_env.step(np.array([0.0]))

    print('New State (get_state): ', test_env.get_state())

    print('New Observation: type: {} | shape: {} | values: {}'.format(type(obs), obs.shape, obs))

    assert test_env.is_valid_state(curr_state) is True
    assert test_env.is_valid_state(None) is False

    test_env.set_state(curr_state)
    print('Reset State (get_state): ', test_env.get_state())

    test_env.plot(blocking=False, save=True, filename='./test/new_obs.png')

    test_env.plot(current_obs, blocking=False, save=True, filename='./test/old_obs_arg.png')


    


should_test = True
if __name__=='__main__':

    if should_test:
        test_file()
    

    

