import numpy as np
import cvxpy as cp

from src.policies.policies import *

class Performative_PredictionV2():

    def __init__(self, env, max_iterations, lamda, reg, gradient, eta, sampling):
        
        self.env = env
        self.max_iterations = max_iterations
        self.lamda = lamda
        self.reg = reg
        self.gradient = gradient
        self.eta = eta
        self.sampling = sampling

        self.reset()

    def reset(self):
        """
        """
        env = self.env
        env.reset()

        self.agent = env.agent
        self.d_diff = []
        self.sub_gap = []

        return

    def execute(self):
        """
        """
        env = self.env    

        self.R, self.T = env._get_RT()
        # initial state action distribution
        d_first = env._get_d(self.T)
        self.d_last = d_first
        for _ in range(self.max_iterations):
            # retrain policy
            self.retrain()
            # update the grid w.r.t. to the agent's policy
            env.update_grid(self.d_last)
            # update rewards and transition functions
            self.R, self.T = env._get_RT()

        return

    def retrain(self):
        """
        """
        env = self.env
        agent = self.agent
        rho = env.rho
        gamma = env.gamma

        # variable
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

        # update policy
        agent.policy = RandomizedD_Policy(agent.actions, d.value)

        return
