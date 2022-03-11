import gym
import gym.spaces
import numpy as np

class MisraAtomEnv(gym.Env):
    def __init__(self,
                 n_factors=10,
                 n_atoms_per_factor=2,
                 horizon=10,
                 noise_std=0.1,
                 permute_actions=True,
                 permute_atoms=True):
        super().__init__()
        self.n_factors = n_factors
        self.n_atoms_per_factor = n_atoms_per_factor
        self.n_atoms = self.n_atoms_per_factor * self.n_factors
        self.n_actions = n_factors
        self.horizon = horizon
        self.noise_std = noise_std

        action_permutation = np.random.permutation if permute_actions else np.arange
        atom_permutation = np.random.permutation if permute_atoms else np.arange

        self.observation_space = gym.spaces.Box(-1, 2, (self.n_atoms, ))
        self.obs_permutations = [atom_permutation(self.n_atoms) for _ in range(self.horizon)]

        self.action_space = gym.spaces.Discrete(self.n_factors)
        self.action_permutations = [
            action_permutation(self.n_actions) for _ in range(self.horizon)
        ]

        self.state = np.empty((self.n_factors, ), dtype=int)
        self.timestep = 0

    def reset(self):
        self.state = np.random.choice(2, (self.n_factors, ))
        self.timestep = 0
        return self.generate_obs(self.state)

    def step(self, action):
        # For each timestep, we define a fixed permutation p of {1,...,d}
        # Dynamics at time step t are given by:
        #   1. Use the action number a to compute the index to update, i = p[a]
        #   2. Set state element s[i] to (1 - s[i])
        #   3. Leave the remaining elements unchanged
        action_permutation = self.action_permutations[self.timestep]
        update_idx = action_permutation[action]
        self.state[update_idx] = 1 - self.state[update_idx]

        obs = self.generate_obs(self.state)
        reward = 0
        self.timestep += 1
        done = (self.timestep >= self.horizon - 1)
        info = self._get_current_info()

        return obs, reward, done, info

    def generate_obs(self, state):
        # Vector z_i = [1, 0] if s[i] = 0, else [0, 1]
        atoms = np.take(np.eye(2), state, axis=0)

        # Sample a scalar Gaussian noise g_i, and add it to both components of z_i
        noise = np.random.normal(loc=0, scale=self.noise_std, size=(self.n_factors, 1))
        atoms += noise

        # Atoms from each factor are then concatentated to generate a vector z \in R^{2d}
        z = np.concatenate(atoms)

        # Apply fixed time-dependent permutation to z to shuffle atoms from different
        # factors. This ensures that an algorithm cannot figure out the children
        # function by relying on the order in which atoms are presented.
        obs_permutation = self.obs_permutations[self.timestep]
        x = np.take(z, obs_permutation)

        return x

    def _get_current_info(self):
        return {
            'timestep': self.timestep,
            'state': self.state,
            'action_permutation': self.action_permutations[self.timestep - 1],
            'obs_permutation': self.obs_permutations[self.timestep - 1]
        }

def test():
    env = MisraAtomEnv(n_factors=5)
    ob = env.reset()
    info = env._get_current_info()

    print_state(None, ob, info)

    for i in range(10):
        a = env.action_space.sample()
        ob, r, done, info = env.step(a)
        print_state(a, ob, info)

def print_state(prev_action, ob, info):
    if prev_action is not None:
        print('  a:', prev_action)
        print('  σ:', info['action_permutation'])
        print('  i:', info['action_permutation'][prev_action])
        print()
    print('t:', info['timestep'])
    print('s:', info['state'])
    print('p:', info['obs_permutation'])
    print('o:', np.around(ob, 2))
    print()

if __name__ == "__main__":
    test()
