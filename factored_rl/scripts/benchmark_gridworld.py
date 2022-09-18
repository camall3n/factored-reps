import numpy as np

from visgrid.envs import GridworldEnv
from visgrid.wrappers.transforms import InvertWrapper, GrayscaleWrapper, wrap_gridworld

from time import time

N = 1000 # steps per trial

#%%
start = time()
env = GridworldEnv(
    rows=10,
    cols=10,
    exploring_starts=True,
    terminate_on_goal=False,
    fixed_goal=True,
    hidden_goal=True,
    agent_position=(5, 3),
    goal_position=(4, 0),
    should_render=False,
)
env.reset()
for _ in range(N):
    action = env.action_space.sample()
    env.step(action)
print(f'Base steps/sec: {N / (time() - start)}')

#%%
start = time()
env = GridworldEnv(10,
                   10,
                   exploring_starts=True,
                   terminate_on_goal=True,
                   fixed_goal=True,
                   hidden_goal=True,
                   should_render=True,
                   dimensions=GridworldEnv.dimensions_onehot)
env = InvertWrapper(GrayscaleWrapper(env))
env.reset()
for _ in range(N):
    action = env.action_space.sample()
    env.step(action)
print(f'Render steps/sec: {N / (time() - start)}')
