import copy
import numpy as np
import cvxpy as cp

from src.envs.gridworld import Gridworld
from src.policies.policies import *

class Performative_Prediction():

    def __init__(self, env: Gridworld, max_iterations, lamda, reg, gradient, eta, sampling, n_sample, policy_gradient, nu, unregularized_obj, lagrangian):
        
        self.env = env
        self.max_iterations = max_iterations
        self.lamda = lamda
        self.reg = reg
        self.gradient = gradient
        self.eta = eta
        self.sampling = sampling
        self.n_sample = n_sample
        self.policy_gradient = policy_gradient
        self.nu = nu
        self.unregularized_obj = unregularized_obj
        self.lagrangian = lagrangian
        # TODO parameter
        # number of rounds for lagrangian method
        self.N = 10
        # parameter delta for lagrangian (beta in doccument)
        self.delta = .1

        self.reset()

    def reset(self):
        """
        """
        env = self.env
        env.reset()

        self.agents = env.agents
        self.d_diff = []
        self.sub_gap = []

        return

    def execute(self):
        """
        """
        env = self.env

        self.R, self.T = env._get_RT()
        # initial state action distribution
        d_first = env._get_d(self.T, self.agents[1])
        self.d_last = d_first
        # initial policy array (needed for policy gradient)
        pi_first = env._get_policy_array(self.agents[1])
        self.pi_last = pi_first
        for _ in range(self.max_iterations):
            # retrain policies
            self.retrain1()
            self.retrain2()
            # update rewards and transition functions
            self.R, self.T = env._get_RT()

        return

    def retrain1(self):
        """
        """
        # different retraining methods
        if self.policy_gradient:
            self.retrain1_policy_gradient()
            return
        elif self.lagrangian:
            self.retrain1_lagrangian()
            return

        env = self.env
        agent = self.agents[1]
        rho = env.rho
        gamma = env.gamma

        # variables
        d = cp.Variable((env.dim, len(agent.actions)), nonneg=True)

        # optimization objective
        if self.gradient:
            target = (1 - self.eta * self.lamda) * self.d_last + self.eta * self.R
            objective = cp.Minimize(cp.power(cp.pnorm(d - target, 2), 2))
        elif self.reg == 'L2':
            objective = cp.Maximize(cp.sum(cp.multiply(d, self.R)) - self.lamda/2 * cp.power(cp.pnorm(d, 2), 2))
        elif self.reg == 'ER':
            objective = cp.Maximize(cp.sum(cp.multiply(d, self.R)) + self.lamda * cp.sum(cp.entr(d)))
        else:
            raise ValueError("Wrong regularizer is given.")

        # constraints
        constraints = []
        for s in env.state_ids:
            if env.is_terminal(s): continue
            constraints.append(cp.sum(d[s]) == rho[s] + gamma * cp.sum(cp.multiply(d, self.T[:,:,s])))

        # solve problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, eps=1e-5)
        
        # store difference in state-action occupancy measure
        self.d_diff.append(np.linalg.norm(d.value - self.d_last)/np.linalg.norm(self.d_last))
        self.d_last = d.value

        # compute suboptimality gap
        if self.gradient:
            # variable
            opt_d = cp.Variable((env.dim, len(agent.actions)), nonneg=True)
            # optimization objective
            opt_objective = cp.Maximize(cp.sum(cp.multiply(opt_d, self.R)) - self.lamda/2 * cp.power(cp.pnorm(opt_d, 2), 2))
            # constraints
            opt_constraints = []
            for s in env.state_ids:
                if env.is_terminal(s): continue
                opt_constraints.append(cp.sum(opt_d[s]) == rho[s] + gamma * cp.sum(cp.multiply(opt_d, self.T[:,:,s])))
            # solve problem
            opt_problem = cp.Problem(opt_objective, opt_constraints)
            opt_problem.solve(solver=cp.SCS, eps=1e-5)
            # suboptimal value
            subopt_problem = cp.sum(cp.multiply(d, self.R)) - self.lamda/2 * cp.power(cp.pnorm(d, 2), 2)
            # store suboptimality gap
            self.sub_gap.append(max((opt_problem.value - subopt_problem.value)/abs(opt_problem.value), 0))   # max0 due to tolerance of SCS
        
        if self.unregularized_obj:
            # variable
            opt_d = cp.Variable((env.dim, len(agent.actions)), nonneg=True)
            # unregularized optimization objective
            opt_objective = cp.Maximize(cp.sum(cp.multiply(opt_d, self.R)))
            # constraints
            opt_constraints = []
            for s in env.state_ids:
                if env.is_terminal(s): continue
                opt_constraints.append(cp.sum(opt_d[s]) == rho[s] + gamma * cp.sum(cp.multiply(opt_d, self.T[:,:,s])))
            # solve problem
            opt_problem = cp.Problem(opt_objective, opt_constraints)
            opt_problem.solve(solver=cp.SCS, eps=1e-5)
            # suboptimal value
            subopt_problem = cp.sum(cp.multiply(d, self.R))
            # store suboptimality gap
            self.sub_gap.append(max((opt_problem.value - subopt_problem.value)/abs(opt_problem.value), 0))   # max0 due to tolerance of SCS

        # update policy
        agent.policy = RandomizedD_Policy(agent.actions, d.value)

        return       

    def retrain2(self):
        """
        """
        env = self.env
        agent = self.agents[2]

        # update policy
        agent.policy = env.response_model(self.agents)
        
        return

    # policy gradient
    def retrain1_policy_gradient(self):
        """
        """
        env = self.env
        agent = self.agents[1]
        fixed_agent = self.agents[2]
        rho = env.rho
        gamma = env.gamma
        
        # compute the derivative of the value function
        d = env._get_d(self.T, agent)
        U = env._get_mU(agent, fixed_agent)
        Q = env._get_mQ(U, agent, fixed_agent)
        DU = np.zeros(shape=(env.dim, len(agent.actions)), dtype='float64')
        for s in env.state_ids:
            for a in agent.actions:
                DU[s, a] = np.sum(d[s]) * Q[s,a]
        
        # variables
        pi = cp.Variable((env.dim, len(agent.actions)), nonneg=True)

        # optimization objective
        target = self.pi_last - self.eta * DU - self.nu * (1 + np.log(self.pi_last))
        objective = cp.Minimize(cp.power(cp.pnorm(pi - target, 2), 2))

        # constraints
        constraints = []
        for s in env.state_ids:
            constraints.append(cp.sum(pi[s]) == 1.0)

        # solve problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, eps=1e-5)

        # modify pi
        delta = 1e-7
        fpi = np.zeros(shape=(env.dim, len(agent.actions)), dtype='float64')
        for s in env.state_ids:
            for a in agent.actions:
                fpi[s, a] = (pi.value[s, a] + delta)/(np.sum(pi.value[s]) + len(agent.actions) * delta)

        # update policy
        agent.policy = Tabular(agent.actions, fpi)
        self.pi_last = copy.deepcopy(fpi)

        # store difference in state-action occupancy measure
        d = env._get_d(self.T, agent)
        self.d_diff.append(np.linalg.norm(d - self.d_last)/np.linalg.norm(self.d_last))
        self.d_last = copy.deepcopy(d)

        # compute suboptimality gap
        # variable
        opt_d = cp.Variable((env.dim, len(agent.actions)), nonneg=True)
        # optimization objective
        opt_objective = cp.Maximize(cp.sum(cp.multiply(opt_d, self.R)) - self.lamda/2 * cp.power(cp.pnorm(opt_d, 2), 2))
        # constraints
        opt_constraints = []
        for s in env.state_ids:
            if env.is_terminal(s): continue
            opt_constraints.append(cp.sum(opt_d[s]) == rho[s] + gamma * cp.sum(cp.multiply(opt_d, self.T[:,:,s])))
        # solve problem
        opt_problem = cp.Problem(opt_objective, opt_constraints)
        opt_problem.solve(solver=cp.SCS, eps=1e-5)
        # suboptimal value
        subopt_value = np.sum(np.multiply(d, self.R)) - self.lamda/2 * np.power(np.linalg.norm(d), 2)
        # store suboptimality gap
        self.sub_gap.append(max((opt_problem.value - subopt_value)/abs(opt_problem.value), 0))   # max0 due to tolerance of SCS

        return

    # policy gradient
    def retrain1_lagrangian(self):
        """
        """
        env = self.env
        agent = self.agents[1]
        rho = env.rho
        gamma = env.gamma

        # get approximate d
        d_hat = env._get_d(self.T, agent)
        # generate empirical data
        data = []
        for _ in range(self.n_sample):
            data += env.sample_trajectory()
        m = len(data)
        # list that contains values d from all the iterates
        d_lst = []
        for n in range(self.N):
            # h Player

            # variables
            h = cp.Variable(env.dim)

            # compute vector L
            L = []
            for s in range(env.dim):
                if env.is_terminal(s): # TODO think and ask
                    L.append(0)
                    continue
                l = rho[s]
                if n==0:
                    L.append(l)
                    continue
                for s_i, a, s_pr, _  in data:
                    if s_i == s:
                        for n_pr in range(n-1):
                            l -= d_lst[n_pr][s, a]/(d_hat[s, a] * m * (1 - gamma))
                    if s_pr == s:
                        for n_pr in range(n-1):
                            l += gamma * (d_lst[n_pr][s, a]/(d_hat[s, a] * m * (1 - gamma)))
                L.append(l)
            print(L)
            print()

            # optimization objective
            objective = cp.Minimize(cp.sum(cp.multiply(L, h)) + self.delta * cp.power(cp.pnorm(h, 2), 2))

            # constraints
            constraints = []
            # ||h||_2 <= 3S/(1-\gamma)^2
            constraints.append(cp.pnorm(h, 2) <= 3 * env.dim / cp.power((1 - gamma), 2))

            # solve problem
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.SCS, eps=1e-5)

            h_t = h.value
            print(h_t)
            print()

            # d Player

            # variables
            d = cp.Variable((env.dim, len(agent.actions)), nonneg=True)

            # optimization objective
            obj = 0
            for s_i, a, s_pr, r  in data:
                obj += d[s, a] * (r - h_t[s] + gamma * h_t[s_pr])/(d_hat[s, a] * m * (1 - gamma))
            objective = cp.Maximize(-self.lamda/2 * cp.power(cp.pnorm(d, 2), 2) + cp.sum(cp.multiply(h.value, rho)) + obj)

            # constraints
            constraints = []
            for s in range(env.dim):
                if env.is_terminal(s): continue
                constraints.append(cp.sum(d[s]) == rho[s] + gamma * cp.sum(cp.multiply(d, self.T[:,:,s])))

            # solve problem
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.SCS, eps=1e-5)

            print(d.value)
            print()

            d_lst.append(d.value)

        # compute average d
        d_avg = np.mean(d_lst, axis=0)

        print(d_avg)
        print()
        
        # store difference in state-action occupancy measure
        self.d_diff.append(np.linalg.norm(d_avg - self.d_last)/np.linalg.norm(self.d_last))
        self.d_last = d_avg

        # update policy
        agent.policy = RandomizedD_Policy(agent.actions, d_avg)

        # 
        print(env._get_policy_array(agent))
        print()
        exit()

        return