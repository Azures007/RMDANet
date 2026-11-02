from copy import deepcopy

import torch
from timm.models.layers import DropPath
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.dropout import _DropoutNd


class EMATeacher(nn.Module):

    def __init__(self, model, classifier, alpha, pseudo_label_weight):
        super(EMATeacher, self).__init__()
        self.ema_model = deepcopy(model)
        self.f = deepcopy(classifier)
        self.alpha = alpha
        self.pseudo_label_weight = pseudo_label_weight
        if self.pseudo_label_weight == 'None':
            self.pseudo_label_weight = None

    @staticmethod
    def _copy_buffers(ema_module, module):
        for ema_buffer, buffer in zip(ema_module.buffers(), module.buffers()):
            ema_buffer.data.copy_(buffer.data)

    @staticmethod
    def _update_ema_buffers(ema_module, module, alpha_teacher):
        for ema_buffer, buffer in zip(ema_module.buffers(), module.buffers()):
            if torch.is_floating_point(ema_buffer):
                ema_buffer.data.mul_(alpha_teacher).add_(buffer.data, alpha=1.0 - alpha_teacher)
            else:
                ema_buffer.data.copy_(buffer.data)

    @staticmethod
    def _set_teacher_eval_layers(module):
        # Keep the parent classifier in training mode so ResClassifier still returns
        # (logits, features), but disable stochastic layers and BN batch-stat updates.
        for m in module.modules():
            if isinstance(m, (_DropoutNd, DropPath, _BatchNorm)):
                m.training = False

    def _init_ema_weights(self, model, classifier):
        # ema model
        for param in self.ema_model.parameters():
            param.detach_()
        mp = list(model.parameters())
        mcp = list(self.ema_model.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()
        self._copy_buffers(self.ema_model, model)
        # f
        for param in self.f.parameters():
            param.detach_()
        mp = list(classifier.parameters())
        mcp = list(self.f.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()
        self._copy_buffers(self.f, classifier)

    def _update_ema(self, model, classifier, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(),
                                    model.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]
        self._update_ema_buffers(self.ema_model, model, alpha_teacher)
        for ema_param, param in zip(self.f.parameters(),
                                    classifier.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]
        self._update_ema_buffers(self.f, classifier, alpha_teacher)

    def update_weights(self, model, classifier, iter):
        # Init/update ema model
        if iter == 0:
            self._init_ema_weights(model, classifier)
        if iter > 0:
            self._update_ema(model, classifier, iter)

    @torch.no_grad()
    def forward(self, target_img):
        # Generate pseudo-label
        self.ema_model.training = False
        self.f.training = True
        self._set_teacher_eval_layers(self.ema_model)
        self._set_teacher_eval_layers(self.f)
        features, projections = self.ema_model(target_img)
        logits, _ = self.f(features)

        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)

        if self.pseudo_label_weight is None:
            pseudo_weight = torch.tensor(1., device=logits.device)
        elif self.pseudo_label_weight == 'prob':
            pseudo_weight = pseudo_prob
        elif self.pseudo_label_weight == 'entropy':
            # 计算预测熵，熵高的权重小，熵低的权重大
            entropy = -torch.sum(ema_softmax * torch.log(ema_softmax + 1e-7), dim=1)
            # 归一化熵到 [0, 1] 范围
            max_entropy = torch.log(torch.tensor(ema_softmax.shape[1], device=logits.device))
            normalized_entropy = entropy / max_entropy
            # 使用 1 - 归一化熵作为权重
            pseudo_weight = 1.0 - normalized_entropy ** 2
        else:
            raise NotImplementedError(self.pseudo_label_weight)

        return pseudo_label, pseudo_weight, features, projections
