"""Learnable supervision budget (SB-B).

Each class owns a learnable threshold tau_c.  A target sample's budget is
    b_i = sigmoid((p_i - tau_{c_i}) / temperature)
which is a soft, differentiable version of the hard confidence gate used by the
current M1 selection.  tau_c is optimised online so that the average budget
matches a preset coverage target (no dataset-specific hand-tuned thresholds).
The budget then acts as the per-sample supervision weight; the M1 loss form and
the reliability operators stay untouched.
"""

from __future__ import annotations

import torch


class LearnableBudget:
    def __init__(self, num_classes, initial_tau=0.95, tau_lr=0.01, coverage=0.6,
                 temperature=0.02, device=None):
        self.coverage = float(coverage)
        self.temperature = float(max(temperature, 1e-4))
        self.tau = torch.nn.Parameter(
            torch.full((int(num_classes),), float(initial_tau), device=device)
        )
        self.optimizer = torch.optim.Adam([self.tau], lr=float(tau_lr))

    def budgets(self, probabilities, labels):
        return torch.sigmoid(
            (probabilities - self.tau[labels.long()]) / self.temperature
        )

    def step(self, probabilities, labels, min_tau=0.50, max_tau=0.999):
        """One coverage-calibration step; returns the detached budgets."""
        budget = self.budgets(probabilities, labels)
        coverage_loss = (budget.mean() - self.coverage) ** 2
        self.optimizer.zero_grad()
        coverage_loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self.tau.clamp_(min_tau, max_tau)
        return budget.detach(), float(budget.mean().detach())
