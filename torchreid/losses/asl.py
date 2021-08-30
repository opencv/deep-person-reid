import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=0,
                    probability_margin=0.05, eps=1e-8,
                    label_smooth=False, ls_eps=0.1):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.label_smooth = label_smooth
        self.ls_eps = ls_eps
        self.clip = probability_margin
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def get_last_scale(self):
        return 1.

    def forward(self, inputs, targets, aug_index=None, lam=None, scale=None, iteration=None):
        """"
        Parameters
        ----------
        inputs: input logits
        targets: targets (multi-label binarized vector)
        """
        if self.label_smooth:
            targets = targets * (1-self.ls_eps)
            targets[targets == 0] = self.ls_eps

        self.targets = targets
        self.anti_targets = 1 - targets

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(inputs)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic BCE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            self.loss *= self.asymmetric_w

        # sum reduction over batch
        return -self.loss.sum()


class AMBinaryLoss(nn.Module):
    def __init__(self, m=0.35, k=0.8, t=1, s=30,
                eps=1e-8, sym_adjustment=False, auto_balance=False,
                label_smooth=False, ls_eps=0.1, gamma_neg=0, gamma_pos=0):
        super().__init__()
        self.sym_adjustment = sym_adjustment
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.auto_balance = auto_balance
        self.eps = eps
        self.label_smooth = label_smooth
        self.ls_eps = ls_eps
        self.m = m
        self.t = t
        self.k = k
        self.s=s

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def get_last_scale(self):
        return self.s

    def sym_adjust(self, z):
        return 2 * torch.pow((z + 1)/2, self.t) - 1

    def forward(self, cos_theta, targets, aug_index=None, lam=None, scale=None, iteration=None):
        """"
        Parameters
        ----------
        cos_theta: dot product between normalized features and proxies
        targets: targets (multi-label binarized vector)
        """
        if self.label_smooth:
            targets = targets * (1-self.ls_eps)
            targets[targets == 0] = self.ls_eps

        self.targets = targets
        self.anti_targets = 1 - targets

        if self.sym_adjustment:
            cos_theta = self.sym_adjust(cos_theta)

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(self.s * (cos_theta - self.m))
        self.xs_neg = torch.sigmoid(-self.s * (cos_theta + self.m))

        if self.auto_balance:
            K = self.targets.size(1)
            C = self.targets.sum(1) # number of target classes for each sample
            balance_koeff_pos = (K - C) / K # balance loss
            balance_koeff_neg = 1 - balance_koeff_pos
        else:
            balance_koeff_pos = self.k / self.s
            balance_koeff_neg = (1 - self.k) / self.s

        self.loss = balance_koeff_pos * self.targets * torch.log(self.xs_pos)
        self.loss.add_(balance_koeff_neg * self.anti_targets * torch.log(self.xs_neg))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            self.xs_pos *= self.targets
            self.xs_neg *= self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            self.loss *= self.asymmetric_w
        # mean reduction over batch
        return self.loss.mean()
