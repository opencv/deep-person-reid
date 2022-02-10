import torch
from torch import nn


class AsymmetricLoss(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=0,
                    probability_margin=0.05, eps=1e-8,
                    label_smooth=0.):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.label_smooth = label_smooth
        self.clip = probability_margin
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, inputs, targets, scale=1., aug_index=None, lam=None):
        """"
        Parameters
        ----------
        inputs: input logits
        targets: targets (multi-label binarized vector)
        """
        if self.label_smooth > 0:
            targets = targets * (1-self.label_smooth)
            targets[targets == 0] = self.label_smooth

        self.anti_targets = 1 - targets

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(scale * inputs)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic BCE calculation
        self.loss = targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            self.xs_pos = self.xs_pos * targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * targets + self.gamma_neg * self.anti_targets)
            self.loss *= self.asymmetric_w

        # sum reduction over batch
        return - self.loss.sum()


class AMBinaryLoss(nn.Module):
    def __init__(self, m=0.35, k=0.8, t=1, s=30,
                eps=1e-8, label_smooth=0., gamma_neg=0,
                gamma_pos=0, probability_margin=0.05):

        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.asymmetric_focus = gamma_neg > 0 or gamma_pos > 0
        self.eps = eps
        self.label_smooth = label_smooth
        self.m = m
        self.k = k
        self.s = s
        self.clip = probability_margin

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, cos_theta, targets, scale=None, aug_index=None, lam=None):
        """"
        Parameters
        ----------
        cos_theta: dot product between normalized features and proxies
        targets: targets (multi-label binarized vector)
        """
        self.s = scale if scale else self.s
        if self.label_smooth > 0:
            targets = targets * (1 - self.label_smooth)
            targets[targets == 0] = self.label_smooth
        self.anti_targets = 1 - targets

        # Calculating Probabilities
        xs_pos = torch.sigmoid(self.s * (cos_theta - self.m))
        xs_neg = torch.sigmoid(- self.s * (cos_theta + self.m))

        if self.asymmetric_focus:
            # Asymmetric Probability Shifting
            if self.clip is not None and self.clip > 0:
                xs_neg = (xs_neg + self.clip).clamp(max=1)
            pt0 = xs_neg * targets
            pt1 = xs_pos * (1 - targets)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            # P_pos ** gamm_neg * (...) + P_neg ** gamma_pos * (...)
            one_sided_w = torch.pow(pt, one_sided_gamma)
            balance_koeff_pos = self.k / self.s
            balance_koeff_neg = (1 - self.k) / self.s
        else:
            assert not self.asymmetric_focus
            # SphereFace2 balancing coefficients
            balance_koeff_pos = self.k / self.s
            balance_koeff_neg = (1 - self.k) / self.s

        self.loss = balance_koeff_pos * targets * torch.log(xs_pos)
        self.loss.add_(balance_koeff_neg * self.anti_targets * torch.log(xs_neg))

        if self.asymmetric_focus:
            self.loss *= one_sided_w

        return - self.loss.sum()
