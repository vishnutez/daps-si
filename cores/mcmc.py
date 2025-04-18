import tqdm
import torch
import numpy as np
import torch.nn as nn
from .trajectory import Trajectory

from side_info_module.face_detection import FaceRecognition


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

    def __init__(self, num_steps, lr, num_steps_fs=None, tau=0.01, lr_min_ratio=0.01, rho=0.005, prior_solver='gaussian', prior_sigma_min=1e-2,
                 mc_algo='langevin', momentum=0.9, **kwargs):
        super().__init__()
        self.num_steps = num_steps
        self.num_steps_fs = num_steps_fs
        self.lr = lr
        self.tau = tau
        self.rho = rho
        self.lr_min_ratio = lr_min_ratio
        self.prior_solver = prior_solver
        self.prior_sigma_min = prior_sigma_min
        self.mc_algo = mc_algo
        self.momentum = momentum
        self.use_face_similarity = kwargs.get('use_face_similarity', False)
        self.norm_order_fs = kwargs.get('norm_order_fs', 2)
        # self.l1_reg = kwargs.get('l1_reg', 0.0)
        # rho is the face similarity sd
        # norm_order_fs == 2, Gaussian is modelled and so we need to divide by it gives 2 * rho^2
        # norm_order_fs == 1, Laplacian is modelled with parameter b = rho / sqrt(2)
        self.scale_fs = 2 * self.rho ** 2 if self.norm_order_fs == 2 else self.rho / np.sqrt(2)
        if self.use_face_similarity:
            print('Using face similarity as side information')
            x_tilde = kwargs.get('x_tilde', None)
            if x_tilde is None:
                raise ValueError("Guidance images (x_tilde) must be provided when using face_similarity is True")
            self.face_recognition = FaceRecognition(mtcnn_face=True, norm_order=self.norm_order_fs)
            self.face_recognition.cuda()
            self.x_tilde = x_tilde.cuda()
            print('x_tilde shape:', x_tilde.shape)

    def score_fn(self, x, x0hat, model, xt, operator, measurement, sigma, fs=False):
        """
        Computes the conditional score function \nabla_x \log p(x_0 = x | x_t, y).

        Returns:
            Tuple containing:
                - Current score estimate.
                - Data-fitting loss.
        """
        data_fitting_grad, data_fitting_loss = operator.gradient(x, measurement, return_loss=True)
        data_term = -data_fitting_grad / self.tau ** 2
        xt_term = (xt - x) / sigma ** 2
        prior_term = self.get_prior_score(x, x0hat, xt, model, sigma)

        # # # Compute the fft of the x term
        # x_freq = torch.fft.fft2(x, dim=(-2, -1), norm='backward')

        # x_freq_real = x_freq.real
        # x_freq_imag = x_freq.imag

        # # Take gradient of l1 norm of x_freq_real, x_freq_imag 

        # l1_real_grad = torch.sgn(x_freq_real)
        # l1_imag_grad = torch.sgn(x_freq_imag)
        # l1_freq_grad = l1_real_grad + l1_imag_grad
        # l1_freq_grad = torch.fft.ifft2(l1_freq_grad, dim=(-2, -1), norm='backward')
        # l1_img_grad = torch.real(l1_freq_grad) + torch.imag(l1_freq_grad) 

        # print('l1_img_grad:', l1_img_grad.max().item())

        # data_term -= self.l1_reg * l1_img_grad  # Take the gradient of the l1 norm of the image

        # print('l1_reg_grad:', l1_reg_grad)

        if self.use_face_similarity and fs:
            face_fitting_loss, face_fitting_grad = self.face_recognition.compute_loss_and_gradient(x, self.x_tilde)           
            face_term = - face_fitting_grad / self.scale_fs   # Norm order = 2, gives 2 * rho^2, Norm order = 1, gives rho 

            # print('data term:', data_term.max().item())
            # print('xt term:', xt_term.max().item())
            # print('prior term:', prior_term.max().item())
            # print('face term:', face_term.max().item()) 

            return data_term + xt_term + prior_term + face_term, data_fitting_loss + face_fitting_loss
        
        return data_term + xt_term + prior_term, data_fitting_loss

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

    def mc_update(self, x, cur_score, lr, epsilon, fs=False):
        """ Performs a single Monte Carlo update step (Langevin or HMC)."""
        if self.mc_algo == 'langevin':
            x_new = x + lr * cur_score + np.sqrt(2 * lr) * epsilon
            # print('No noise added in Langevin sampling')
            # x_new = x + lr * cur_score
        elif self.mc_algo == 'hmc':  # (damping) hamiltonian monte carlo
            step_size = np.sqrt(lr)
            self.velocity = self.momentum * self.velocity + step_size * cur_score + np.sqrt(2 * (1 - self.momentum)) * epsilon
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
            log_accept_prob= log_data_ratio + log_prior_ratio
            accept = torch.rand_like(log_accept_prob).log() < log_accept_prob
            accept = accept.view(-1, *[1] * len(x.shape[1:]))
            # update: accept new sample
            x = torch.where(accept, x_new, x)
        return x.detach()

    def sample(self, xt, model, x0hat, operator, measurement, sigma, ratio, record=False, verbose=False):
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

        x = x0hat.clone().detach()
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        step = 0
        freq = self.num_steps // self.num_steps_fs if self.num_steps_fs is not None else 1
        for _ in pbar:
            step += 1
            if step % freq == 0 and self.num_steps_fs is not None:
                cur_score, fitting_loss = self.score_fn(x, x0hat, model, xt, operator, measurement, sigma, fs=True)
            else:
                cur_score, fitting_loss = self.score_fn(x, x0hat, model, xt, operator, measurement, sigma, fs=False)
            epsilon = torch.randn_like(x)

            x = self.mc_update(x, cur_score, lr, epsilon)

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
