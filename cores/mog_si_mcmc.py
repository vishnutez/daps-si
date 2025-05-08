import tqdm
import torch
import numpy as np
import torch.nn as nn
from .trajectory import Trajectory


class MCMCSampler(nn.Module):
    """
    Monte Carlo sampler class for diffusion processes.

    Supports Langevin dynamics, Hamiltonian Monte Carlo (HMC) and Metropolis-Hastings (MH) methods.

    Attributes:
        num_steps (int): Number of sampling steps.
        lr (float): Initial learning rate.
        tau (float): Standard deviation for data-fitting term.
        lr_min_ratio (float): Minimum learning rate ratio.
        prior_solver (str): Method to compute prior score ('gaussian', 'exact', 'score-t', 'score-min').
        prior_sigma_min (float): Minimum sigma for prior computation.
        mc_algo (str): Monte Carlo algorithm ('langevin', 'mh', or 'hmc').
        momentum (float): Momentum coefficient for HMC.
    """

    def __init__(self, num_steps, lr, tau=0.01, lr_min_ratio=0.01, prior_solver='gaussian', prior_sigma_min=1e-2,
                 mc_algo='langevin', momentum=0.9, sigma_thres=0, mult_init=True):
        super().__init__()
        self.num_steps = num_steps
        self.lr = lr
        self.tau = tau
        self.lr_min_ratio = lr_min_ratio
        self.prior_solver = prior_solver
        self.prior_sigma_min = prior_sigma_min
        self.mc_algo = mc_algo
        self.momentum = momentum
        self.sigma_thres = sigma_thres
        self.mult_init = mult_init

    def score_fn(self, x, x0hat, model, xt, operator, measurement, sigma, gradient_rewards, mc_step):
        """
        Computes the conditional score function \nabla_x \log p(x_0 = x | x_t, y).

        Returns:
            Tuple containing:
                - Current score estimate.
                - Data-fitting loss.
        """
        num_particles = x0hat.shape[0]
        # print(f'num_particles: {num_particles}')
        y = measurement[0].unsqueeze(0)
        xt_term = (xt - x) / sigma ** 2
        prior_term = self.get_prior_score(x, x0hat, xt, model, sigma)

        delta = (x-x0hat).reshape(num_particles, -1)
        mog_dist = -torch.linalg.norm(delta, axis=-1) ** 2 / (2 ** sigma ** 2)
        mog_weight = torch.softmax(mog_dist, dim=0)
        mog_term = torch.sum(mog_weight.reshape(-1, 1, 1, 1) * (xt_term + prior_term), dim=0)

        # print(f'x: {x.shape}')
        # print(f'y: {y.shape}')
        # print(f'mog_dist: {mog_dist}')
        # print(f'mog_weight: {mog_weight}')
        # print(f"mog_term: {mog_term.shape}")

        # here we add the gradient of the rewards
        rewards_grad_term = torch.zeros_like(x, device=x.device)
        data_term = torch.zeros_like(x, device=x.device)
        data_fitting_loss = 0
        for reward in gradient_rewards:
            if mc_step % reward.freq == 0:
                if reward.name == 'measurement':
                    data_fitting_grad, data_fitting_loss = reward.get_gradients(x, measurements=y)
                    data_term = -data_fitting_grad / self.tau ** 2
                else:
                    rewards_grad_term += reward.scale * reward.get_gradients(x)

        rewards_grad_term = -rewards_grad_term / self.tau ** 2

        return data_term + mog_term + rewards_grad_term, data_fitting_loss

    def get_prior_score(self, x, x0hat, xt, model, sigma):
        if self.prior_solver == 'score-min' or self.prior_solver == 'score-t' or self.prior_solver == 'gaussian':
            prior_score = self.prior_score
        elif self.prior_solver == 'exact':
            prior_score = model.score(x, torch.tensor(self.prior_sigma_min).to(x.device)).detach()
        else:
            raise NotImplementedError
        return prior_score

    def prepare_prior_score(self, x0hat, xt, model, sigma):
        """
        Precomputes the prior score based on the specified solver method.
        """
        if self.prior_solver == 'score-min':
            self.prior_score = model.score(x0hat, self.prior_sigma_min).detach()

        elif self.prior_solver == 'score-t':
            self.prior_score = model.score(xt, sigma).detach()

        elif self.prior_solver == 'gaussian':
            self.prior_score = (x0hat - xt).detach() / sigma ** 2

        elif self.prior_solver == 'exact':
            pass

        else:
            raise NotImplementedError

    def mc_prepare(self, x0hat, xt, model, operator, measurement, sigma):
        """Prepares the sampler state before starting Monte Carlo sampling."""
        if self.mc_algo == 'hmc':
            self.velocity = torch.randn_like(x0hat)

    def mc_update(self, x, cur_score, lr, epsilon):
        """ Performs a single Monte Carlo update step (Langevin or HMC)."""
        if self.mc_algo == 'langevin':
            x_new = x + lr * cur_score + np.sqrt(2 * lr) * epsilon
        elif self.mc_algo == 'hmc':  # (damping) hamiltonian monte carlo
            step_size = np.sqrt(lr)
            self.velocity = self.momentum * self.velocity + step_size * cur_score + np.sqrt(
                2 * (1 - self.momentum)) * epsilon
            x_new = x + self.velocity * step_size
        else:
            raise NotImplementedError
        return x_new

    def sample_mh(self, xt, model, x0hat, operator, measurement, sigma, ratio, record=False, verbose=False):
        if record:
            self.trajectory = Trajectory()

        lr = self.get_lr(ratio)
        x = x0hat.clone().detach()
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        for _ in pbar:
            # Gaussain proposal
            x_new = x + torch.randn_like(x) * np.sqrt(lr)
            # compute acceptance ratio
            loss_new = operator.loss(x_new, measurement)
            loss_old = operator.loss(x, measurement)
            log_data_ratio = (loss_old - loss_new) / (2 * self.tau ** 2)
            # compute prior p(x_0 | x_t)
            prior_loss_new = (x_new - x0hat).pow(2).flatten(1).sum(-1)
            prior_loss_old = (x - x0hat).pow(2).flatten(1).sum(-1)
            log_prior_ratio = (prior_loss_old - prior_loss_new) / (2 * sigma ** 2)
            # compute acceptance probability
            log_accept_prob = log_data_ratio + log_prior_ratio
            accept = torch.rand_like(log_accept_prob).log() < log_accept_prob
            accept = accept.view(-1, *[1] * len(x.shape[1:]))
            # update: accept new sample
            x = torch.where(accept, x_new, x)
        return x.detach()

    def sample(self, xt, model, x0hat, operator, measurement, sigma, ratio, gradient_rewards, record=False, verbose=False, x_init=None):
        """
        Main method for performing MCMC sampling.

        Args:
            xt (torch.Tensor): Current noisy latent tensor.
            model (nn.Module): Diffusion model providing the score function.
            x0hat (torch.Tensor): Initial estimate of x0 from PF-ODE.
            operator (Operator): Measurement operator.
            measurement (torch.Tensor): Measurement data.
            sigma (float): Noise scale at current timestep.
            ratio (float): Ratio to adjust learning rate scheduling.
            gradient_rewards (list): rewards that should be used as guidance in MCMC
            record (bool): Whether to record trajectory.
            verbose (bool): Verbosity flag.

        Returns:
            torch.Tensor: Sampled latent tensor.
        """
        if self.mc_algo == 'mh':
            return self.sample_mh(xt, model, x0hat, operator, measurement, sigma, ratio, record, verbose)
        if record:
            self.trajectory = Trajectory()
        lr = self.get_lr(ratio)
        self.mc_prepare(x0hat, xt, model, operator, measurement, sigma)
        self.prepare_prior_score(x0hat, xt, model, sigma)

        # if x_init is None:
        #     x_init = x0hat.clone().detach()
        #     if self.mult_init:
        #         x = x_init
        #     elif sigma > self.sigma_thres:
        #         print(f'sigma: {sigma} sigma_thres: {self.sigma_thres} => init with mean')
        #         x = torch.mean(x_init, dim=0, keepdim=True) # initialization
        #     else:
        #         print(f'sigma: {sigma} sigma_thres: {self.sigma_thres} => init with random mode')
        #         x = x_init[0].unsqueeze(0) # initialization 
        # else:
        #     x = x_init.clone().detach() 
        #     print(f'init with x_init') 
        
        if self.mult_init: 
            x_init = x0hat.clone().detach()
            x = x_init.clone().detach()
            # print(f'init with x_init: ', x.shape)
        else:
            x_init = torch.mean(x0hat.clone().detach(), dim=0, keepdim=True)
            x = x_init.clone().detach()

        num_particles = x.shape[0]
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        for mc_step in pbar:
            
            if not self.mult_init:
                cur_score, fitting_loss = self.score_fn(x, x0hat, model, xt, operator, measurement, sigma, gradient_rewards, mc_step)
                epsilon = torch.randn_like(x)
                x = self.mc_update(x, cur_score, lr, epsilon)
            else:
                # print(f'Updating with multiple initializations')
                for n in range(num_particles):
                    xn = x[n].unsqueeze(0)
                    cur_score, fitting_loss = self.score_fn(xn, x0hat, model, xt, operator, measurement, sigma, gradient_rewards, mc_step)
                    epsn = torch.randn_like(xn)
                    x_updated = self.mc_update(xn, cur_score, lr, epsn)
                    # print(f'x_updated: {x_updated.shape}')
                    x[n] = x_updated

            # early stopping with NaN
            if torch.isnan(x).any():
                return torch.zeros_like(x)

                # record
            if record:
                self._record(x, epsilon, fitting_loss.sqrt())
        return x.detach()

    def _record(self, x, epsilon, loss):
        """
            Records the intermediate states during sampling.
        """
        self.trajectory.add_tensor(f'xi', x)
        self.trajectory.add_tensor(f'epsilon', epsilon)
        self.trajectory.add_value(f'loss', loss)

    def get_lr(self, ratio):
        """
            Computes the learning rate based on the given ratio.
        """
        p = 1
        multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr

    def summary(self):
        print('+' * 50)
        print('MCMC Sampler Summary')
        print('+' * 50)
        print(f"Prior Solver    : {self.prior_solver}")
        print(f"MCMC Algorithm  : {self.mc_algo}")
        print(f"Num Steps       : {self.num_steps}")
        print(f"Learning Rate   : {self.lr}")
        print(f"Tau             : {self.tau}")
        print(f"LR Min Ratio    : {self.lr_min_ratio}")
        print('+' * 50)
