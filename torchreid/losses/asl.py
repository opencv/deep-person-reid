import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, probability_margin=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = probability_margin
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def get_last_scale(self):
        return 1.

    def forward(self, inputs, targets, aug_index=None, lam=None, scale=None, iteration=None):
        """"
        Parameters
        ----------
        inputs: input logits
        targets: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(inputs)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Probability Shifting
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class AsymmetricAMSoftmax(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, m=0.35, k=0.8, t=1, s=30, eps=1e-8, sym_adjustment=False):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.sym_adjustment = sym_adjustment
        self.eps = eps
        self.m = m
        self.t = t
        self.k = k
        self.s=s

    def get_last_scale(self):
        return 1.

    def sym_adjust(self, z):
        return 2 * torch.pow((z + 1)/2, self.t) - 1

    def forward(self, cos_theta, targets, aug_index=None, lam=None, scale=None, iteration=None):
        """"
        Parameters
        ----------
        cos_theta: dot product between normalized features and proxies
        targets: targets (multi-label binarized vector)
        """
        # TO DO maybe balance loss automatically based on information about number of target classes
        # K = targets.size(1) # class number
        # self.k = (K - 1) / K # balance loss
        if self.sym_adjustment:
            cos_theta = self.sym_adjust(cos_theta)
        Lpos = (self.k / self.s) * targets * torch.log(1 + torch.exp(-self.s * (cos_theta - self.m)))
        Lneg = (1 - self.k)/self.s * (1 - targets) * torch.log(1 + torch.exp(self.s * (cos_theta + self.m)))
        loss = Lpos + Lneg

        # TO DO combine assym loss with AMSoftmax
        # if self.gamma_neg > 0 or self.gamma_pos > 0:
        #     if self.disable_torch_grad_focal_loss:
        #         torch.set_grad_enabled(False)
        #     pt0 = Lpos * targets
        #     pt1 = Lneg * (1 - targets)  # pt = p if t > 0 else 1-p
        #     pt = pt0 + pt1
        #     one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        #     one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        #     if self.disable_torch_grad_focal_loss:
        #         torch.set_grad_enabled(True)
        #     loss *= one_sided_w

        return loss.sum()

class AsymmetricLossOptimized(nn.Module):
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
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def get_last_scale(self):
        return 1.

    def forward(self, inputs, targets):
        """"
        Parameters
        ----------
        inputs: input logits
        targets: targets (multi-label binarized vector)
        """
        if self.label_smooth > 0:
            targets = targets * (1-self.label_smooth)
            targets[targets == 0] = self.label_smooth

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
                label_smooth=0., gamma_neg=0, gamma_pos=0):
        super().__init__()
        self.sym_adjustment = sym_adjustment
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.asymmetric_focus = gamma_neg > 0 or gamma_pos > 0
        self.auto_balance = auto_balance
        self.eps = eps
        self.label_smooth = label_smooth
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

    def forward(self, cos_theta, targets):
        """"
        Parameters
        ----------
        cos_theta: dot product between normalized features and proxies
        targets: targets (multi-label binarized vector)
        """
        if self.label_smooth > 0:
            targets = targets * (1-self.label_smooth)
            targets[targets == 0] = self.label_smooth

        self.targets = targets
        self.anti_targets = 1 - targets

        if self.sym_adjustment:
            cos_theta = self.sym_adjust(cos_theta)

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(self.s * (cos_theta - self.m))
        self.xs_neg = torch.sigmoid(self.s * (-cos_theta - self.m))

        if self.auto_balance:
            assert not self.asymmetric_focus, "Auto balance is not compatible with asymmetric focussing"
            K = self.targets.size(1)
            C = self.targets.sum(1, keepdim=True) # number of target classes for each sample
            balance_koeff_pos = (K - C) / K # balance loss
            balance_koeff_neg = 1 - balance_koeff_pos
        elif self.asymmetric_focus:
            balance_koeff_pos = 1
            balance_koeff_neg = 1
        else:
            assert not self.asymmetric_focus and not self.auto_balance
            balance_koeff_pos = self.k / self.s
            balance_koeff_neg = (1 - self.k) / self.s
        self.loss = balance_koeff_pos * self.targets * torch.log(1 + torch.exp(-self.s * (cos_theta - self.m)))
        self.loss.add_(balance_koeff_neg * self.anti_targets * torch.log(1 + torch.exp(self.s * (cos_theta + self.m))))

        # Asymmetric Focusing
        if self.asymmetric_focus:
            self.xs_pos *= self.targets
            self.xs_neg *= self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            self.loss *= self.asymmetric_w

        return self.loss.sum()
