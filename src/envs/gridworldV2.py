import numpy as np
from numpy.testing import assert_almost_equal
import copy
import itertools

from src.agents.gw_agent import Agent
from src.policies.policies import *


class GridworldV2():

    def __init__(self, beta, eps, gamma, sampling=False, n_sample=500, seed=1, max_sample_steps=100):

        # random generator
        self.rng = np.random.default_rng(seed)
        # grid
        h = -0.5
        f = -0.02
        self.h = h
        self.f = f
        self.t = -0.01
        self.initial_grid = np.array(
            [[-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01],
             [-0.01, -0.01, f, -0.01, h, -0.01, -0.01, -0.01],
             [-0.01, -0.01, -0.01, h, -0.01, -0.01, f, -0.01],
             [-0.01, f, -0.01, -0.01, -0.01, h, -0.01, f],
             [-0.01, -0.01, -0.01, h, -0.01, -0.01, f, -0.01],
             [-0.01, h, h, -0.01, f, -0.01, h, -0.01],
             [-0.01, h, -0.01, -0.01, h, -0.01, h, -0.01],
             [-0.01, -0.01, -0.01, h, -0.01, f, -0.01, +1]])
        self.grid_shape = self.initial_grid.shape
        self.dim = np.prod(self.grid_shape)
        self.state_ids = range(self.dim)
        self.terminal_state_id = self.dim - 1
        self.reward_space = [-0.01, f, h, 1]
        # initial state distribution
        self.initial_states = [s for s in self.state_ids if (state := self._get_state(s)) and (state[0] == 0 or state[1] == 0)]
        self.rho = self._get_initial_state_distribution()
        # move: left, right, up, down
        self.moves = [0, 1, 2, 3]
        self.move_mapping = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        # agent
        self.action_space = self.moves
        self.agent = Agent(1, self.action_space)
        # stochastic transition dynamics
        self.eps = eps
        self.T = self.get_T()
        # parameters
        self.beta = beta
        assert 3 * self.beta <= self.dim - len(self.initial_states) - 1, "Parameter beta is too big."
        self.gamma = gamma
        # sampling
        self.sampling = sampling
        self.n_sample = n_sample
        self.max_sample_steps = max_sample_steps
        
        self.reset()

    def reset(self):
        """
        """
        self.grid = self.initial_grid
        self.initialize_policy()
        d = self._get_d(self.T)
        # update the grid w.r.t. to the agent's policy
        self.update_grid(d)

        return

    def get_next_state(self, state_id, move):
        """
        Returns next state when transitions are derministic
        """
        assert move in self.moves, f"Illegal move {move} was given."

        state = self._get_state(state_id)

        if self.is_terminal(state_id):
            return state_id

        next_state = self.vector_add(state, self.move_mapping[move])

        if not self.is_valid(next_state):
            next_state = state

        next_state_id = self._get_state_id(next_state)

        return next_state_id

    def get_reward(self, state_id, next_state_id):
        """
        Returns reward
        """ 
        if self.is_terminal(state_id):
            reward = 0
            return reward

        state = self._get_state(next_state_id)
        reward = self.grid[state[0], state[1]]

        return reward

    def step(self, s, a):
        """
        """
        # next state
        r_states = self._get_reachable_states(s)
        probs = [self.T[s, a, s_pr] for s_pr in r_states]
        s_pr = self.rng.choice(r_states, p=probs)
        # reward
        r = self.get_reward(s, s_pr)

        return s_pr, r

    def value_iteration(self, R, T, tol=1e-5):
        """
        """
        gamma = self.gamma

        # initialize all state values to zero
        U = np.zeros(self.dim, dtype='float64')
        while True:
            U_old = copy.deepcopy(U)
            delta = 0
            for s in self.state_ids:
                r_states = self._get_reachable_states(s)
                if self.is_terminal(s): 
                    U[s] = 0
                else:
                    U[s] = np.max([
                        R[s, a] + gamma * np.sum([T[s, a, s_pr] * U_old[s_pr] for s_pr in r_states])
                        for a in self.action_space
                    ])
                delta = max(delta, abs(U[s] - U_old[s]))

            if delta < tol:
                break

        return U

    def _get_Q(self, R, T, U):
        """
        """
        gamma = self.gamma

        # initialize all state-action values to zero
        Q = np.zeros(shape=(self.dim, len(self.action_space)), dtype='float64')
        for s in self.state_ids:
            r_states = self._get_reachable_states(s)
            for a in self.action_space:
                if self.is_terminal(s):
                    Q[s, a] = 0
                else:
                    Q[s, a] = R[s, a] + gamma * np.sum([T[s, a, s_pr] * U[s_pr] for s_pr in r_states])

        return Q

    def _get_initial_state_distribution(self):
        """
        """
        rho = np.zeros(self.dim, dtype='float64')
        initial_states = self.initial_states
        for s in initial_states:
            rho[s] = 1 / len(initial_states)

        return rho

    # performative prediction
    def _get_RT(self):
        """
        Get R and T functions
        """
        if not self.sampling:
            return self._get_exact_RT()
        else:
            return self._get_approximate_RT()

    def _get_exact_RT(self):
        """
        """
        T = self.T

        R = np.zeros(shape=(self.dim, len(self.action_space)), dtype='float64')
        for s in self.state_ids:
            r_states = self._get_reachable_states(s)
            for a in self.action_space:
                for r_s in r_states:
                    r = self.get_reward(s, r_s)
                    R[s, a] += T[s, a, r_s] * r

        return R, T

    def _get_approximate_RT(self):
        """
        """
        agent = self.agent
        rho = self.rho
        n_sample = self.n_sample
        rng = self.rng

        # total values
        T_tot = np.zeros(shape=(self.dim, len(self.action_space), self.dim), dtype='float64')
        R_tot = np.zeros(shape=(self.dim, len(self.action_space)), dtype='float64')
        # visitattion count
        V = np.zeros(shape=(self.dim, len(self.action_space)), dtype='int')
        # begin sampling trajectories
        for _ in range(n_sample):
            n_steps = 0
            # initial state
            s = rng.choice(np.arange(self.dim), p=rho)
            while not self.is_terminal(s) and n_steps < self.max_sample_steps:
                # action
                a = agent.take_action(s, rng)
                # update visitation count
                V[s, a] += 1
                # next state and reward
                s_pr, r = self.step(s, a)
                # update total values
                T_tot[s, a, s_pr] += 1
                R_tot[s, a] += r
                # prepare for next time-step
                s = s_pr
                n_steps += 1
        
        # approximated values
        T_hat = np.zeros(shape=(self.dim, len(self.action_space), self.dim), dtype='float64')
        R_hat = np.zeros(shape=(self.dim, len(self.action_space)), dtype='float64')
        # remove +1 from reward space
        reward_space = [r for r in self.reward_space if r != 1]
        for s in self.state_ids:
            r_states = self._get_reachable_states(s)
            for a in self.action_space:
                if self.is_terminal(s):
                    R_hat[s, a] = 0
                    T_hat[s, a, self.terminal_state_id] = 1
                elif V[s, a]:
                    R_hat[s, a] = R_tot[s, a] / V[s, a]
                    for s_pr in r_states:
                        T_hat[s, a, s_pr] = T_tot[s, a, s_pr] / V[s, a]
                else:
                    # if s, a not seen during sampling assign to R[s,a] the mean of all negative rewards
                    R_hat[s, a] = np.mean(reward_space)
                    # if if s, a not seen during sampling assign to T[s, a] the uniform distiribution over all reachable states
                    for s_pr in r_states:
                        T_hat[s, a, s_pr] = 1 / len(r_states)

                assert_almost_equal(np.sum(T_hat[s, a, :]), 1)

        return R_hat, T_hat

    def update_grid(self, d):
        """
        """
        beta = self.beta
        gamma = self.gamma

        # get state occupancy
        v = np.array([np.sum(d[s, a] for a in self.action_space) for s in self.state_ids])
        # make state occupancy of initial and terminal states -1 so that they don't get to be picked next
        for s in self.initial_states:
            v[s] = -1
        v[self.terminal_state_id] = -1
        # assign h to the beta states with the highest occupancy
        ind = np.argpartition(v, range(-1, -beta - 1, -1))[- beta:]
        for s in ind:
            state = self._get_state(s)
            self.grid[state[0]][state[1]] = self.h
        # assign f to the beta states with the second highest occupancy
        ind = np.argpartition(v, range(-beta - 1, -2*beta - 1, -1))[- 2*beta:-beta]
        for s in ind:
            state = self._get_state(s)
            self.grid[state[0]][state[1]] = self.f
        # make state occupancy of initial and terminal states 1/(1-\gamma) so that they don't get to be picked next
        for s in self.initial_states:
            v[s] = 1/(1-gamma)
        v[self.terminal_state_id] = 1/(1-gamma)
        # assign the time-step cost t to the beta states with the lowest occupancy
        ind = np.argpartition(v, range(beta))[:beta]
        for s in ind:
            state = self._get_state(s)
            self.grid[state[0]][state[1]] = self.t

        return

    def _get_d(self, T, tol=1e-5):
        """
        """
        agent = self.agent
        gamma = self.gamma
        rho = self.rho

        d = np.zeros(shape=(self.dim, len(self.action_space)), dtype='float64')
        while True:
            d_old = copy.deepcopy(d)
            delta = 0
            for s, a in itertools.product(self.state_ids, self.action_space):
                if self.is_terminal(s):
                    d[s, a] = 0
                else:
                    p = agent._get_probs(s)
                    d[s, a] = p[a] * (rho[s] + gamma * np.sum(d_old * T[:,:,s]))
                delta = max(delta, abs(d[s, a] - d_old[s, a])) 
            if delta < tol:
                break

        # sanity check
        for s in self.state_ids[:-1]:
            assert_almost_equal(np.sum(d[s]), rho[s] + gamma * np.sum(d * T[:,:,s]), decimal=5)
            for a in self.action_space:
                assert d[s, a] >= 0, f"Actual: {d[s, a]}"

        return d


    # utilities
    def is_valid(self, state):
        """
        """
        grid_shape = self.grid_shape

        return bool(0 <= state[0] < grid_shape[0] and 0 <= state[1] < grid_shape[1])

    def is_terminal(self, state_id):
        """
        """

        return bool(state_id == self.terminal_state_id)

    def _get_state(self, state_id):
        """
        """
        grid_shape = self.grid_shape
        state = (state_id // grid_shape[0], state_id % grid_shape[1])

        return state

    def _get_state_id(self, state):
        """
        """
        grid_shape = self.grid_shape
        state_id = state[0] * grid_shape[0] + state[1]

        return state_id

    def _get_reachable_states(self, state_id):
        """
        """
        r_states = []

        for move in self.moves:
            next_state_id = self.get_next_state(state_id, move)
            if next_state_id not in r_states: r_states.append(next_state_id)

        return r_states

    def get_T(self):
        """
        Gets exact transition dynamics of the environment (need to be computed only once in the begining)
        """
        T = np.zeros(shape=(self.dim, len(self.action_space), self.dim), dtype='float64')
        for s in self.state_ids:
            r_states = self._get_reachable_states(s)
            for a in self.action_space:
                s_pr = self.get_next_state(s, a)
                T[s, a, s_pr] = 1 - self.eps
                for r_s in r_states:
                    T[s, a, r_s] += self.eps / len(r_states)

                assert_almost_equal(np.sum(T[s, a, :]), 1)

        return T

    def initialize_policy(self):
        """
        """
        agent = self.agent
        # sampling policy is random
        R, T = self._get_RT()

        V_star = self.value_iteration(R, T)
        Q_star = self._get_Q(R, T, V_star)
        agent.policy = Eps_Greedy_Policy(self.action_space, Q_star)

        return


    @staticmethod
    def vector_add(x, y):
        return (x[0] + y[0], x[1] + y[1])
