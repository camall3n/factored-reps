import numpy as np

from visgrid.envs import GridworldEnv
from visgrid.wrappers.sensors import *
from visgrid.wrappers.transforms import wrap_gridworld

from time import time

N = 1000 # steps per trial

#%%
start = time()
env = GridworldEnv(
    rows=6,
    cols=6,
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
env = GridworldEnv(rows=6,
                   cols=6,
                   exploring_starts=True,
                   terminate_on_goal=False,
                   fixed_goal=True,
                   hidden_goal=True,
                   agent_position=(5, 3),
                   goal_position=(4, 0),
                   should_render=True,
                   dimensions=GridworldEnv.dimensions_6x6_to_18x18)
env = wrap_gridworld(env)
env.reset()
for _ in range(N):
    action = env.action_space.sample()
    env.step(action)
print(f'Sensor steps/sec: {N / (time() - start)}')

#%%
start = time()
env = GridworldEnv(
    rows=6,
    cols=6,
    exploring_starts=True,
    terminate_on_goal=False,
    fixed_goal=True,
    hidden_goal=True,
    agent_position=(5, 3),
    goal_position=(4, 0),
    should_render=True,
)
env.reset()
for _ in range(N):
    action = env.action_space.sample()
    env.step(action)
print(f'Rendered steps/sec: {N / (time() - start)}')
