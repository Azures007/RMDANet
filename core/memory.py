from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ReliabilityBatch:
    labels: torch.Tensor
    confidence: torch.Tensor
    margin: torch.Tensor
    reliability: torch.Tensor
    stable_mask: torch.Tensor
    agreement: torch.Tensor
    feature_drift: torch.Tensor
    spatial_probabilities: torch.Tensor
    spatial_score: torch.Tensor
    spatial_support: torch.Tensor
    prototype_agreement: torch.Tensor


class ReliabilityPrototypeMemory:
    """Unified target reliability memory and multi-prototype adaptation.

    A single reliability score controls pseudo-label selection, target
    sub-prototype updates, source-target anchor fusion and target loss weights.
    """

    def __init__(
        self,
        num_samples: int,
        num_classes: int,
        feature_dim: int,
        neighbor_ids: torch.Tensor,
        device: torch.device,
        probability_momentum: float = 0.92,
        feature_momentum: float = 0.94,
        target_prior_momentum: float = 0.985,
        num_subprototypes: int = 4,
        source_prototype_momentum: float = 0.97,
        target_prototype_update_rate: float = 0.16,
        target_prototype_create_similarity: float = 0.90,
        reliability_threshold: float = 0.22,
        pseudo_confidence: float = 0.78,
        pseudo_margin: float = 0.055,
        class_cap: int = 18,
        min_age: int = 2,
        min_streak: int = 2,
        use_probability_history: bool = True,
        use_time_score: bool = True,
        use_feature_score: bool = True,
        use_view_score: bool = True,
        use_guard: bool = True,
        use_prototype_veto: bool = True,
        use_spatial: bool = True,
        use_target_prototypes: bool = True,
        reliability_weighted_prototype_update: bool = True,
        adaptive_anchor: bool = True,
        fixed_target_anchor_weight: float = 0.50,
        strict_class_balance_strength: float = 0.0,
        collapse_guard: bool = False,
    ) -> None:
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.device = device
        self.probability_momentum = float(probability_momentum)
        self.feature_momentum = float(feature_momentum)
        self.target_prior_momentum = float(target_prior_momentum)
        self.source_prototype_momentum = float(source_prototype_momentum)
        self.target_prototype_update_rate = float(target_prototype_update_rate)
        self.target_prototype_create_similarity = float(target_prototype_create_similarity)
        self.reliability_threshold = float(reliability_threshold)
        self.pseudo_confidence = float(pseudo_confidence)
        self.pseudo_margin = float(pseudo_margin)
        self.class_cap = int(class_cap)
        self.min_age = int(min_age)
        self.min_streak = int(min_streak)
        self.use_probability_history = bool(use_probability_history)
        self.use_time_score = bool(use_time_score)
        self.use_feature_score = bool(use_feature_score)
        self.use_view_score = bool(use_view_score)
        self.use_guard = bool(use_guard)
        self.use_prototype_veto = bool(use_prototype_veto)
        self.use_spatial = bool(use_spatial)
        self.use_target_prototypes = bool(use_target_prototypes)
        self.reliability_weighted_prototype_update = bool(
            reliability_weighted_prototype_update
        )
        self.adaptive_anchor = bool(adaptive_anchor)
        self.strict_class_balance_strength = float(min(max(strict_class_balance_strength, 0.0), 1.0))
        self.collapse_guard = bool(collapse_guard)
        self.fixed_target_anchor_weight = float(
            min(max(fixed_target_anchor_weight, 0.0), 1.0)
        )

        self.prob_memory = torch.zeros(num_samples, num_classes, device=device)
        self.feature_memory = torch.zeros(num_samples, feature_dim, device=device)
        self.valid = torch.zeros(num_samples, dtype=torch.bool, device=device)
        self.age = torch.zeros(num_samples, dtype=torch.long, device=device)
        self.streak = torch.zeros(num_samples, dtype=torch.long, device=device)
        self.last_label = torch.full((num_samples,), -1, dtype=torch.long, device=device)
        self.label_flip_count = torch.zeros(num_samples, dtype=torch.long, device=device)
        self.reliability_memory = torch.zeros(num_samples, device=device)
        self.clean_mask = torch.ones(num_samples, dtype=torch.bool, device=device)
        self.cleanlab_disabled = False
        self.target_prior = torch.full((num_classes,), 1.0 / num_classes, device=device)
        self.prior_updates = 0
        # Running distribution of accepted pseudo labels. This is used only by
        # the opt-in collapse guard and never by the original Houston path.
        self.selection_prior = torch.full((num_classes,), 1.0 / num_classes, device=device)
        self.selection_updates = 0

        neighbors = neighbor_ids.long().to(device)
        if neighbors.ndim != 2 or neighbors.size(0) != num_samples:
            raise ValueError("neighbor_ids must have shape [num_samples, max_neighbors].")
        self.neighbor_ids = neighbors

        self.source_prototypes = torch.zeros(num_classes, feature_dim, device=device)
        self.source_valid = torch.zeros(num_classes, dtype=torch.bool, device=device)
        self.source_support = torch.zeros(num_classes, device=device)

        self.num_subprototypes = max(1, int(num_subprototypes))
        shape = (num_classes, self.num_subprototypes)
        self.target_prototypes = torch.zeros(*shape, feature_dim, device=device)
        self.target_valid = torch.zeros(*shape, dtype=torch.bool, device=device)
        self.target_support = torch.zeros(*shape, device=device)
        self.target_reliability = torch.zeros(*shape, device=device)
        self.target_updates = torch.zeros(*shape, device=device)
        self.target_stability = torch.ones(*shape, device=device)

    @torch.no_grad()
    def state_dict(self) -> dict[str, torch.Tensor | int | float]:
        return {
            "prob_memory": self.prob_memory.detach().cpu(),
            "feature_memory": self.feature_memory.detach().cpu(),
            "valid": self.valid.detach().cpu(),
            "age": self.age.detach().cpu(),
            "streak": self.streak.detach().cpu(),
            "last_label": self.last_label.detach().cpu(),
            "reliability_memory": self.reliability_memory.detach().cpu(),
            "target_prior": self.target_prior.detach().cpu(),
            "prior_updates": int(self.prior_updates),
            "selection_prior": self.selection_prior.detach().cpu(),
            "selection_updates": int(self.selection_updates),
            "source_prototypes": self.source_prototypes.detach().cpu(),
            "source_valid": self.source_valid.detach().cpu(),
            "source_support": self.source_support.detach().cpu(),
            "target_prototypes": self.target_prototypes.detach().cpu(),
            "target_valid": self.target_valid.detach().cpu(),
            "target_support": self.target_support.detach().cpu(),
            "target_reliability": self.target_reliability.detach().cpu(),
            "target_updates": self.target_updates.detach().cpu(),
            "target_stability": self.target_stability.detach().cpu(),
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, torch.Tensor | int | float]) -> None:
        tensor_names = [
            "prob_memory",
            "feature_memory",
            "valid",
            "age",
            "streak",
            "last_label",
            "reliability_memory",
            "target_prior",
            "source_prototypes",
            "source_valid",
            "source_support",
            "target_prototypes",
            "target_valid",
            "target_support",
            "target_reliability",
            "target_updates",
            "target_stability",
        ]
        for name in tensor_names:
            destination = getattr(self, name)
            destination.copy_(state[name].to(self.device))
        self.prior_updates = int(state.get("prior_updates", 0))
        if "selection_prior" in state:
            self.selection_prior.copy_(state["selection_prior"].to(self.device))
        self.selection_updates = int(state.get("selection_updates", 0))

    @torch.no_grad()
    def update_source_prototypes(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        features = F.normalize(features.detach().float(), dim=1)
        labels = labels.detach().long()
        for class_id in range(self.num_classes):
            mask = labels.eq(class_id)
            if not mask.any():
                continue
            prototype = F.normalize(features[mask].mean(dim=0), dim=0)
            if not bool(self.source_valid[class_id].item()):
                self.source_prototypes[class_id] = prototype
                self.source_valid[class_id] = True
            else:
                updated = (
                    self.source_prototype_momentum * self.source_prototypes[class_id]
                    + (1.0 - self.source_prototype_momentum) * prototype
                )
                self.source_prototypes[class_id] = F.normalize(updated, dim=0)
            self.source_support[class_id] += float(mask.sum().item())


    @torch.no_grad()
    def refine_probabilities(
        self,
        probabilities: torch.Tensor,
        features: torch.Tensor,
        temperature: float = 0.09,
        max_blend: float = 0.12,
    ) -> torch.Tensor:
        """Reliability-bounded classifier/prototype probability fusion."""
        probabilities = probabilities.detach().float().clamp_min(1e-8)
        features = F.normalize(features.detach().float(), dim=1)
        anchors = torch.zeros(
            self.num_classes, self.feature_dim, device=self.device, dtype=torch.float32
        )
        valid = self.source_valid.clone()
        for class_id in range(self.num_classes):
            if not bool(self.source_valid[class_id].item()):
                continue
            source_anchor = F.normalize(self.source_prototypes[class_id].float(), dim=0)
            target_mask = self.target_valid[class_id] if self.use_target_prototypes else torch.zeros_like(self.target_valid[class_id])
            if target_mask.any():
                target_slots = self.target_prototypes[class_id, target_mask].float()
                support = self.target_support[class_id, target_mask]
                reliability = self.target_reliability[class_id, target_mask].clamp(0.0, 1.0)
                stability = self.target_stability[class_id, target_mask].clamp(0.0, 1.0)
                weights = (support.sqrt() * reliability * stability).clamp_min(1e-5)
                target_anchor = F.normalize(
                    torch.sum(target_slots * weights[:, None], dim=0) / weights.sum(), dim=0
                )
                maturity = (1.0 - torch.exp(-support.sum() / 20.0)).clamp(0.0, 0.65)
                anchors[class_id] = F.normalize(
                    (1.0 - maturity) * source_anchor + maturity * target_anchor, dim=0
                )
            else:
                anchors[class_id] = source_anchor
        if not valid.any():
            return probabilities
        similarities = features @ F.normalize(anchors, dim=1).t()
        similarities[:, ~valid] = -1e4
        prototype_probability = F.softmax(similarities / max(float(temperature), 1e-4), dim=1)
        classifier_label = probabilities.argmax(dim=1)
        prototype_label = prototype_probability.argmax(dim=1)
        prototype_confidence = prototype_probability.max(dim=1).values
        agreement = classifier_label.eq(prototype_label).float()
        blend = float(max_blend) * torch.sigmoid((prototype_confidence - 0.40) / 0.10)
        blend = blend * agreement
        refined = (1.0 - blend[:, None]) * probabilities + blend[:, None] * prototype_probability
        return refined / refined.sum(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def _update_target_prior(self, probabilities: torch.Tensor) -> None:
        batch_prior = probabilities.detach().float().mean(dim=0)
        batch_prior = batch_prior / batch_prior.sum().clamp_min(1e-8)
        if self.prior_updates == 0:
            self.target_prior.copy_(batch_prior)
        else:
            self.target_prior.mul_(self.target_prior_momentum).add_(
                batch_prior, alpha=1.0 - self.target_prior_momentum
            )
        self.target_prior.div_(self.target_prior.sum().clamp_min(1e-8))
        self.prior_updates += 1

    @torch.no_grad()
    def distribution_align(self, probabilities: torch.Tensor, gamma: float = 0.06) -> torch.Tensor:
        probabilities = probabilities.detach().float().clamp_min(1e-8)
        batch_prior = probabilities.mean(dim=0).clamp_min(1e-8)
        reference = self.target_prior.clamp_min(1e-5)
        ratio = (reference / batch_prior).pow(float(gamma)).clamp(0.75, 1.35)
        aligned = probabilities * ratio[None, :]
        return aligned / aligned.sum(dim=1, keepdim=True).clamp_min(1e-8)

    @torch.no_grad()
    def empty_batch(
        self,
        sample_ids: torch.Tensor,
        probabilities: torch.Tensor,
        features: torch.Tensor,
    ) -> ReliabilityBatch:
        probabilities = probabilities.detach().float().clamp_min(1e-8)
        top = torch.topk(probabilities, k=min(2, self.num_classes), dim=1)
        labels = top.indices[:, 0]
        confidence = top.values[:, 0]
        margin = top.values[:, 0] - top.values[:, 1] if self.num_classes > 1 else torch.ones_like(confidence)
        zeros = confidence.new_zeros(confidence.shape)
        false = torch.zeros_like(labels, dtype=torch.bool)
        return ReliabilityBatch(
            labels=labels,
            confidence=confidence,
            margin=margin,
            reliability=zeros,
            stable_mask=false,
            agreement=false,
            feature_drift=zeros,
            spatial_probabilities=probabilities,
            spatial_score=zeros,
            spatial_support=zeros,
            prototype_agreement=false,
        )

    @torch.no_grad()
    def current_pseudo_batch(
        self,
        sample_ids: torch.Tensor,
        teacher_probabilities: torch.Tensor,
        student_probabilities: torch.Tensor,
        teacher_features: torch.Tensor,
        strong_probabilities: torch.Tensor | None = None,
    ) -> ReliabilityBatch:
        """Mean-teacher pseudo labels without cross-epoch memory or prototypes."""
        teacher_probabilities = teacher_probabilities.detach().float().clamp_min(1e-8)
        student_probabilities = student_probabilities.detach().float().clamp_min(1e-8)
        self._update_target_prior(teacher_probabilities)
        aligned = self.distribution_align(teacher_probabilities)
        top = torch.topk(aligned, k=min(2, self.num_classes), dim=1)
        labels = top.indices[:, 0]
        confidence = top.values[:, 0]
        margin = top.values[:, 0] - top.values[:, 1] if self.num_classes > 1 else torch.ones_like(confidence)
        agreement = labels.eq(student_probabilities.argmax(dim=1))
        if strong_probabilities is not None:
            agreement = agreement & labels.eq(strong_probabilities.detach().argmax(dim=1))
        thresholds = self.class_thresholds()[labels]
        confidence_score = torch.sigmoid((confidence - thresholds) / 0.035)
        margin_score = torch.sigmoid((margin - self.pseudo_margin) / 0.030)
        view_score = 0.10 + 0.90 * agreement.float()
        reliability = (confidence_score * margin_score * view_score + 1e-12).pow(1.0 / 3.0)
        stable = (
            (confidence >= thresholds)
            & (margin >= self.pseudo_margin)
            & agreement
        )
        stable = self._class_balanced_top_mask(labels, reliability, stable)
        stable = stable & self.clean_mask[sample_ids]
        stable = self._collapse_guard_mask(labels, reliability, stable)
        zeros = confidence.new_zeros(confidence.shape)
        return ReliabilityBatch(
            labels=labels,
            confidence=confidence,
            margin=margin,
            reliability=reliability,
            stable_mask=stable,
            agreement=agreement,
            feature_drift=zeros,
            spatial_probabilities=aligned,
            spatial_score=zeros,
            spatial_support=zeros,
            prototype_agreement=torch.ones_like(agreement),
        )

    def class_thresholds(self) -> torch.Tensor:
        prior = self.target_prior.clamp_min(1e-6)
        prior = prior / prior.sum().clamp_min(1e-8)
        uniform = 1.0 / self.num_classes
        relative = torch.log((prior + 1e-6) / uniform)
        s = self.strict_class_balance_strength
        coef = 0.030 + 0.020 * s
        lo = -0.015 + 0.010 * s
        hi = 0.040 + 0.030 * s
        thresholds = self.pseudo_confidence + coef * torch.tanh(relative)
        return thresholds.clamp(self.pseudo_confidence + lo, self.pseudo_confidence + hi)

    @torch.no_grad()
    def _spatial_consensus(
        self,
        sample_ids: torch.Tensor,
        current_features: torch.Tensor,
        fallback: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.neighbor_ids.size(1) == 0:
            zeros = fallback.new_zeros(fallback.size(0))
            return fallback, zeros, zeros
        neighbors = self.neighbor_ids[sample_ids]
        valid_index = neighbors.ge(0)
        safe = neighbors.clamp_min(0)
        valid = valid_index & self.valid[safe]
        support = valid.sum(dim=1).float()
        neighbor_probabilities = self.prob_memory[safe].float()
        neighbor_features = self.feature_memory[safe].float()
        current = F.normalize(current_features.detach().float(), dim=1)
        cosine = torch.sum(neighbor_features * current[:, None, :], dim=2).clamp(-1.0, 1.0)
        weights = torch.sigmoid((cosine - 0.50) / 0.10) * valid.float()
        weight_sum = weights.sum(dim=1, keepdim=True)
        consensus = (neighbor_probabilities * weights[:, :, None]).sum(dim=1)
        consensus = consensus / weight_sum.clamp_min(1e-8)
        no_support = weight_sum.squeeze(1).le(1e-8)
        consensus = torch.where(no_support[:, None], fallback, consensus)
        consensus = consensus / consensus.sum(dim=1, keepdim=True).clamp_min(1e-8)
        consensus_confidence = consensus.max(dim=1).values
        mean_similarity = weight_sum.squeeze(1) / support.clamp_min(1.0)
        spatial_score = (
            (support / 3.0).clamp(0.0, 1.0) * consensus_confidence * mean_similarity
        ).clamp(0.0, 1.0)
        spatial_score = torch.where(no_support, torch.zeros_like(spatial_score), spatial_score)
        return consensus, spatial_score, support

    @torch.no_grad()
    def _source_prototype_evidence(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.source_valid.any():
            return torch.ones(features.size(0), device=features.device), torch.ones(
                features.size(0), dtype=torch.bool, device=features.device
            )
        normalized = F.normalize(features.detach().float(), dim=1)
        prototypes = F.normalize(self.source_prototypes.float(), dim=1)
        similarities = normalized @ prototypes.t()
        similarities[:, ~self.source_valid] = -1e4
        predicted = similarities.argmax(dim=1)
        best = similarities.max(dim=1).values
        agreement = predicted.eq(labels)
        score = torch.sigmoid((best - 0.20) / 0.10)
        score = score * (0.40 + 0.60 * agreement.float())
        return score.clamp(0.0, 1.0), agreement

    def class_conditional_alignment(
        self, target_features: torch.Tensor, temperature: float = 0.10
    ) -> torch.Tensor:
        """Soft-assignment class-conditional alignment.

        For each source class c, compute a soft target prototype by weighting
        target features with their softmax similarity to source prototype c.
        Pull source prototype c and target soft prototype c together, push
        source prototype c away from the most similar wrong-class target proto.
        """
        if not bool(self.source_valid.all().item()):
            return target_features.new_tensor(0.0)
        f = F.normalize(target_features.float(), dim=1)
        protos = F.normalize(self.source_prototypes.float(), dim=1)
        sim = f @ protos.t() / max(float(temperature), 1e-4)   # [N, C]
        soft = F.softmax(sim, dim=1)                            # [N, C]
        tgt_protos = soft.t() @ f                               # [C, D]
        tgt_protos = F.normalize(tgt_protos, dim=1)
        pos = (protos * tgt_protos).sum(dim=1)                  # [C]
        sim_mat = protos @ tgt_protos.t()                       # [C, C]
        eye = torch.eye(self.num_classes, device=sim_mat.device, dtype=torch.bool)
        neg_mat = sim_mat.masked_fill(eye, -10.0)
        neg = neg_mat.max(dim=1).values                         # [C]
        return F.softplus(neg - pos + 0.3).mean()

    @torch.no_grad()
    def update(
        self,
        sample_ids: torch.Tensor,
        teacher_probabilities: torch.Tensor,
        student_probabilities: torch.Tensor,
        teacher_features: torch.Tensor,
        strong_probabilities: torch.Tensor | None = None,
        guard_probabilities: torch.Tensor | None = None,
    ) -> ReliabilityBatch:
        """Update cross-epoch memory and return only conservatively stable samples.

        Labels, confidence and margin come from the updated probability memory,
        not from a single current prediction.  Current teacher, student, strong
        view, frozen semantic guard, source prototype and spatial neighborhood
        are used as independent vetoes.  This prevents excessive pseudo-label coverage from becoming confirmation bias.
        """
        sample_ids = sample_ids.long()
        teacher_probabilities = teacher_probabilities.detach().float().clamp_min(1e-8)
        student_probabilities = student_probabilities.detach().float().clamp_min(1e-8)
        teacher_features = F.normalize(teacher_features.detach().float(), dim=1)
        self._update_target_prior(teacher_probabilities)
        current_aligned = self.distribution_align(teacher_probabilities)
        current_teacher_labels = current_aligned.argmax(dim=1)

        was_valid = self.valid[sample_ids]
        old_features = self.feature_memory[sample_ids].clone()
        old_probabilities = self.prob_memory[sample_ids].clone()
        feature_drift = torch.where(
            was_valid,
            1.0 - torch.sum(old_features * teacher_features, dim=1).clamp(-1.0, 1.0),
            torch.full((sample_ids.numel(),), 0.20, device=self.device),
        )

        updated_probabilities = torch.where(
            was_valid[:, None],
            self.probability_momentum * old_probabilities
            + (1.0 - self.probability_momentum) * current_aligned,
            current_aligned,
        )
        updated_probabilities = updated_probabilities / updated_probabilities.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        updated_features = torch.where(
            was_valid[:, None],
            self.feature_momentum * old_features
            + (1.0 - self.feature_momentum) * teacher_features,
            teacher_features,
        )
        updated_features = F.normalize(updated_features, dim=1)

        probability_basis = updated_probabilities if self.use_probability_history else current_aligned
        history_top = torch.topk(probability_basis, k=min(2, self.num_classes), dim=1)
        confidence = history_top.values[:, 0]
        margin = (
            history_top.values[:, 0] - history_top.values[:, 1]
            if self.num_classes > 1
            else torch.ones_like(confidence)
        )
        labels = history_top.indices[:, 0]

        previous_label = self.last_label[sample_ids]
        new_streak = torch.where(
            was_valid & previous_label.eq(labels),
            self.streak[sample_ids] + 1,
            torch.ones_like(self.streak[sample_ids]),
        )
        prev_label_check = self.last_label[sample_ids]
        flip_mask = was_valid & (prev_label_check >= 0) & prev_label_check.ne(labels)
        self.label_flip_count[sample_ids] += flip_mask.long()
        self.age[sample_ids] += 1
        self.streak[sample_ids] = new_streak
        self.last_label[sample_ids] = labels
        self.prob_memory[sample_ids] = updated_probabilities
        self.feature_memory[sample_ids] = updated_features
        self.valid[sample_ids] = True

        student_labels = student_probabilities.argmax(dim=1)
        agreement = labels.eq(current_teacher_labels) & labels.eq(student_labels)
        if strong_probabilities is not None:
            strong_probabilities = strong_probabilities.detach().float().clamp_min(1e-8)
            agreement = agreement & labels.eq(strong_probabilities.argmax(dim=1))
        if not self.use_view_score:
            agreement = torch.ones_like(agreement)

        if (not self.use_guard) or guard_probabilities is None:
            guard_agreement = torch.ones_like(agreement)
        else:
            guard_probabilities = guard_probabilities.detach().float().clamp_min(1e-8)
            guard_agreement = labels.eq(guard_probabilities.argmax(dim=1))

        if self.use_spatial:
            spatial_probabilities, spatial_score, spatial_support = self._spatial_consensus(
                sample_ids, teacher_features, updated_probabilities
            )
            spatial_labels = spatial_probabilities.argmax(dim=1)
            spatial_agreement = spatial_labels.eq(labels)
        else:
            spatial_probabilities = probability_basis
            spatial_score = confidence.new_zeros(confidence.shape)
            spatial_support = confidence.new_zeros(confidence.shape)
            spatial_agreement = torch.ones_like(agreement)

        if self.use_prototype_veto:
            prototype_score, prototype_agreement = self._source_prototype_evidence(
                teacher_features, labels
            )
        else:
            prototype_score = torch.ones_like(confidence)
            prototype_agreement = torch.ones_like(agreement)

        thresholds = self.class_thresholds()[labels]
        confidence_score = torch.sigmoid((confidence - thresholds) / 0.035)
        margin_score = torch.sigmoid((margin - self.pseudo_margin) / 0.030)
        age_score = (self.age[sample_ids].float() / 7.0).clamp(0.0, 1.0)
        streak_score = (self.streak[sample_ids].float() / 6.0).clamp(0.0, 1.0)
        time_score = torch.sqrt((age_score * streak_score).clamp_min(1e-8))
        drift_score = torch.exp(-feature_drift / 0.14).clamp(0.0, 1.0)
        view_score = 0.10 + 0.90 * agreement.float()
        guard_score = 0.20 + 0.80 * guard_agreement.float()
        spatial_component = torch.where(
            spatial_support.ge(3.0),
            0.20 + 0.80 * spatial_score * spatial_agreement.float(),
            torch.ones_like(spatial_score),
        )
        components = [confidence_score, margin_score]
        if self.use_time_score:
            components.append(time_score)
        if self.use_feature_score:
            components.append(drift_score)
        if self.use_view_score:
            components.append(view_score)
        if self.use_guard:
            components.append(guard_score)
        if self.use_prototype_veto:
            components.append(prototype_score)
        if self.use_spatial:
            components.append(spatial_component)
        product = torch.ones_like(confidence_score)
        for component in components:
            product = product * component
        reliability = (product + 1e-12).pow(1.0 / float(len(components))).clamp(0.0, 1.0)
        self.reliability_memory[sample_ids] = (
            0.82 * self.reliability_memory[sample_ids] + 0.18 * reliability
        )

        exceptional = (confidence >= 0.96) & (margin >= 0.22)
        prototype_veto = prototype_agreement | exceptional
        guard_veto = guard_agreement | (exceptional & prototype_agreement)
        spatial_veto = (
            spatial_support.lt(3.0)
            | (spatial_agreement & spatial_score.ge(0.42))
            | exceptional
        )
        stable = (
            (reliability >= self.reliability_threshold)
            & (confidence >= thresholds)
            & (margin >= self.pseudo_margin)
        )
        if self.use_time_score:
            stable = stable & (self.age[sample_ids] >= self.min_age) & (self.streak[sample_ids] >= self.min_streak)
        if self.use_view_score:
            stable = stable & agreement
        if self.use_prototype_veto:
            stable = stable & prototype_veto
        if self.use_guard:
            stable = stable & guard_veto
        if self.use_spatial:
            stable = stable & spatial_veto
        # Cleanlab 过滤必须在 class-cap 之前，否则噪声样本先占 quota
        stable = stable & self.clean_mask[sample_ids]
        stable = self._class_balanced_top_mask(labels, reliability, stable)
        stable = self._collapse_guard_mask(labels, reliability, stable)
        if stable.any() and self.use_target_prototypes:
            self._update_target_subprototypes(
                teacher_features[stable], labels[stable], reliability[stable]
            )

        return ReliabilityBatch(
            labels=labels,
            confidence=confidence,
            margin=margin,
            reliability=reliability,
            stable_mask=stable,
            agreement=agreement,
            feature_drift=feature_drift,
            spatial_probabilities=spatial_probabilities,
            spatial_score=spatial_score,
            spatial_support=spatial_support,
            prototype_agreement=prototype_agreement,
        )

    @torch.no_grad()
    def flip_diagnostics(self) -> dict:
        """统计每个目标样本的标签翻牌次数。"""
        valid = self.valid
        if not valid.any():
            return {"mean_flips": 0.0, "pct_zero": 0.0, "pct_ge3": 0.0,
                    "pct_ge5": 0.0, "n_valid": 0}
        flips = self.label_flip_count[valid].float()
        return {
            "mean_flips": float(flips.mean().item()),
            "pct_zero": float((flips == 0).float().mean().item()),
            "pct_ge3": float((flips >= 3).float().mean().item()),
            "pct_ge5": float((flips >= 5).float().mean().item()),
            "n_valid": int(valid.sum().item()),
        }

    @torch.no_grad()
    def refresh_clean_mask(self) -> None:
        """Use cleanlab to remove noisy pseudo-labeled target samples."""
        if self.cleanlab_disabled:
            return
        try:
            import numpy as np
            import cleanlab
        except ImportError:
            print("[cleanlab] not installed, skipping")
            self.cleanlab_disabled = True
            return

        valid_idx = self.valid.nonzero(as_tuple=True)[0]
        if len(valid_idx) < 100:
            return

        probs = self.prob_memory[valid_idx].detach().cpu().numpy()
        labs = probs.argmax(axis=1)

        try:
            issues = cleanlab.pruning.get_noise_indices(
                labs, probs, prune_method='prune_by_class'
            )
        except Exception as e:
            print(f"[cleanlab] refresh failed: {e}")
            return

        issue_set = set(int(i) for i in issues.tolist())
        removed = 0
        for local_i, global_i in enumerate(valid_idx.tolist()):
            if local_i in issue_set and bool(self.clean_mask[global_i].item()):
                self.clean_mask[global_i] = False
                removed += 1
        print(f"[cleanlab] valid={len(valid_idx)} issues={len(issue_set)} newly_removed={removed} total_kept={int(self.clean_mask.sum().item())}")

    def _class_balanced_top_mask(
        self,
        labels: torch.Tensor,
        reliability: torch.Tensor,
        candidate: torch.Tensor,
    ) -> torch.Tensor:
        selected = torch.zeros_like(candidate)
        prior = self.target_prior.clamp_min(1e-6)
        uniform = 1.0 / self.num_classes
        for class_id in range(self.num_classes):
            indices = torch.nonzero(candidate & labels.eq(class_id), as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            relative = float((prior[class_id] / uniform).sqrt().clamp(0.85, 1.25).item())
            dynamic_cap = max(3, int(round(self.class_cap * relative)))
            fixed_cap = max(3, int(self.class_cap))
            s = self.strict_class_balance_strength
            cap = int(round((1.0 - s) * dynamic_cap + s * fixed_cap))
            if indices.numel() <= cap:
                selected[indices] = True
                continue
            keep = torch.topk(reliability[indices], k=cap, largest=True).indices
            selected[indices[keep]] = True
        return selected

    @torch.no_grad()
    def _collapse_guard_mask(
        self,
        labels: torch.Tensor,
        reliability: torch.Tensor,
        candidate: torch.Tensor,
    ) -> torch.Tensor:
        """Suppress a pseudo class only when it stays abnormally dominant.

        The guard is label-free: it compares the running distribution of accepted
        pseudo labels with a conservative reference that blends a uniform prior
        and the teacher's running target prior. It does not special-case any class
        id and is disabled by default.
        """
        if not self.collapse_guard or not candidate.any():
            return candidate

        selected = candidate.clone()
        uniform = torch.full_like(self.selection_prior, 1.0 / self.num_classes)
        reference = 0.65 * uniform + 0.35 * self.target_prior.detach().clamp_min(1e-6)
        reference = reference / reference.sum().clamp_min(1e-8)

        # Let the running statistic warm up before it is allowed to veto samples.
        if self.selection_updates >= 10:
            dominance = self.selection_prior.clamp_min(1e-6) / reference.clamp_min(1e-6)
            for class_id in range(self.num_classes):
                ratio = float(dominance[class_id].item())
                if ratio <= 1.45:
                    continue
                indices = torch.nonzero(
                    selected & labels.eq(class_id), as_tuple=False
                ).flatten()
                if indices.numel() == 0:
                    continue

                # A dominant class must be more reliable, but the extra gate is
                # deliberately mild so a genuinely frequent class is not erased.
                extra = min(0.055, 0.050 * math.log(max(ratio, 1.0)))
                floor = min(0.95, self.reliability_threshold + extra)
                indices = indices[reliability[indices] >= floor]
                selected[labels.eq(class_id)] = False
                if indices.numel() == 0:
                    continue

                # Also shrink the per-batch quota smoothly as dominance grows.
                cap = max(3, int(round(self.class_cap / math.sqrt(ratio))))
                if indices.numel() > cap:
                    keep = torch.topk(
                        reliability[indices], k=cap, largest=True
                    ).indices
                    indices = indices[keep]
                selected[indices] = True

        if selected.any():
            counts = torch.bincount(
                labels[selected], minlength=self.num_classes
            ).to(self.selection_prior.dtype)
            distribution = counts / counts.sum().clamp_min(1.0)
            momentum = 0.96
            self.selection_prior.mul_(momentum).add_(
                distribution, alpha=1.0 - momentum
            )
            self.selection_prior.div_(self.selection_prior.sum().clamp_min(1e-8))
            self.selection_updates += 1
        return selected

    @torch.no_grad()
    def _update_target_subprototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        reliability: torch.Tensor,
    ) -> None:
        for feature, label, rel in zip(features, labels, reliability):
            feature = F.normalize(feature.detach().float(), dim=0)
            class_id = int(label.item())
            rel_value = rel.detach().float().clamp(0.0, 1.0)
            valid = self.target_valid[class_id]
            if not valid.any():
                slot = 0
                self.target_prototypes[class_id, slot] = feature
                self.target_valid[class_id, slot] = True
                self.target_support[class_id, slot] = rel_value
                self.target_reliability[class_id, slot] = rel_value
                self.target_updates[class_id, slot] = 1.0
                continue

            similarities = torch.sum(
                self.target_prototypes[class_id].float() * feature[None, :], dim=1
            )
            similarities = similarities.masked_fill(~valid, -1e4)
            best_similarity, best_slot_tensor = similarities.max(dim=0)
            if (~valid).any() and float(best_similarity.item()) < self.target_prototype_create_similarity:
                slot = int(torch.nonzero(~valid, as_tuple=False).flatten()[0].item())
                self.target_prototypes[class_id, slot] = feature
                self.target_valid[class_id, slot] = True
                self.target_support[class_id, slot] = rel_value
                self.target_reliability[class_id, slot] = rel_value
                self.target_updates[class_id, slot] = 1.0
                continue

            slot = int(best_slot_tensor.item())
            old = self.target_prototypes[class_id, slot].clone()
            reliability_factor = float(rel_value.item()) if self.reliability_weighted_prototype_update else 1.0
            update_rate = max(
                1e-4,
                min(0.45, self.target_prototype_update_rate * reliability_factor),
            )
            new = F.normalize((1.0 - update_rate) * old + update_rate * feature, dim=0)
            movement = 1.0 - torch.sum(old * new).clamp(-1.0, 1.0)
            self.target_prototypes[class_id, slot] = new
            self.target_support[class_id, slot] += rel_value
            self.target_reliability[class_id, slot] = (
                0.90 * self.target_reliability[class_id, slot] + 0.10 * rel_value
            )
            self.target_updates[class_id, slot] += 1.0
            current_stability = torch.exp(-movement / 0.05)
            self.target_stability[class_id, slot] = (
                0.90 * self.target_stability[class_id, slot] + 0.10 * current_stability
            )

    def weighted_pseudo_loss(
        self,
        logits: torch.Tensor,
        batch: ReliabilityBatch,
    ) -> torch.Tensor:
        if not batch.stable_mask.any():
            return logits.new_tensor(0.0)
        per_sample = F.cross_entropy(
            logits[batch.stable_mask].float(),
            batch.labels[batch.stable_mask],
            reduction="none",
        )
        weights = batch.reliability[batch.stable_mask].detach().pow(1.5).clamp_min(1e-4)
        if self.strict_class_balance_strength > 0:
            selected_labels = batch.labels[batch.stable_mask]
            class_counts = torch.bincount(
                selected_labels, minlength=self.num_classes
            ).to(weights.dtype)
            normalization = class_counts[selected_labels].clamp_min(1.0).sqrt()
            weights = weights / (normalization ** self.strict_class_balance_strength)

        # === Uncertainty weighting ===
        selected_logits = logits[batch.stable_mask].float()
        probs = F.softmax(selected_logits, dim=1).clamp_min(1e-8)
        entropy = -(probs * probs.log()).sum(dim=1)
        max_entropy = math.log(max(2, self.num_classes))
        norm_entropy = (entropy / max_entropy).clamp(0.0, 1.0)
        uncertainty_weight = torch.exp(-norm_entropy / 0.3).detach()
        weights = weights * uncertainty_weight

        return torch.sum(per_sample * weights) / weights.sum().clamp_min(1e-6)

    def prototype_align_flagment_loss(
        self,
        student_features: torch.Tensor,
        batch: ReliabilityBatch,
        min_confidence: float = 0.95,
        temperature: float = 0.08,
    ) -> torch.Tensor:
        """Pavia-only source-class prototype alignment for high-confidence pseudo labels.

        This deliberately does not alter the existing prototype/CEMPA path. It uses only
        source prototypes and target samples that already passed the cross-epoch reliability
        selection, plus a confidence floor, so the new loss cannot create pseudo labels by itself.
        """
        selected = batch.stable_mask & batch.confidence.ge(float(min_confidence))
        if int(selected.sum().item()) < 2 or not self.source_valid.any():
            return student_features.new_tensor(0.0)

        features = F.normalize(student_features[selected].float(), dim=1)
        labels = batch.labels[selected]
        weights = batch.reliability[selected].detach().clamp_min(1e-5)
        source = F.normalize(self.source_prototypes.float(), dim=1)
        valid_source = self.source_valid

        terms: list[torch.Tensor] = []
        for feature, label, weight in zip(features, labels, weights):
            class_id = int(label.item())
            if not bool(valid_source[class_id].item()):
                continue
            similarities = source @ feature
            positive = similarities[class_id]
            negative_mask = valid_source.clone()
            negative_mask[class_id] = False
            if bool(negative_mask.any().item()):
                negatives = similarities[negative_mask]
                temperature_value = max(float(temperature), 1e-4)
                logits = torch.cat([positive.reshape(1), negatives]) / temperature_value
                target = torch.zeros(1, dtype=torch.long, device=feature.device)
                loss = F.cross_entropy(logits.reshape(1, -1), target)
            else:
                loss = 1.0 - positive
            terms.append(loss * weight)

        if not terms:
            return student_features.new_tensor(0.0)
        return torch.stack(terms).sum() / weights.sum().clamp_min(1e-6)

    def unsupervised_diagnostics(self) -> dict[str, float]:
        valid = self.valid
        if not valid.any():
            return {
                "score": 0.0,
                "reliability": 0.0,
                "coverage": 0.0,
                "entropy": 1.0,
                "prototype_coverage": 0.0,
                "prior_stability": 0.0,
                "pseudo_peak": float(self.selection_prior.max().item()),
            }
        probabilities = self.prob_memory[valid].clamp_min(1e-8)
        labels = probabilities.argmax(dim=1)
        entropy = (
            -(probabilities * probabilities.log()).sum(dim=1).mean()
            / math.log(max(2, self.num_classes))
        )
        histogram = torch.bincount(labels, minlength=self.num_classes).float().to(self.device)
        histogram = histogram / histogram.sum().clamp_min(1e-8)
        active = self.target_prior >= 0.002
        if not active.any():
            active = torch.ones_like(active)
        coverage = (histogram[active] >= 0.001).float().mean()
        prototype_coverage = self.target_valid.any(dim=1)[active].float().mean()
        midpoint = 0.5 * (histogram.clamp_min(1e-8) + self.target_prior.clamp_min(1e-8))
        js = 0.5 * (
            torch.sum(histogram.clamp_min(1e-8) * (histogram.clamp_min(1e-8).log() - midpoint.log()))
            + torch.sum(
                self.target_prior.clamp_min(1e-8)
                * (self.target_prior.clamp_min(1e-8).log() - midpoint.log())
            )
        )
        prior_stability = torch.exp(-js / 0.08)
        reliability = self.reliability_memory[valid].mean()
        score = (
            0.42 * reliability
            + 0.20 * (1.0 - entropy)
            + 0.14 * coverage
            + 0.14 * prototype_coverage
            + 0.10 * prior_stability
        )
        return {
            "score": float(score.item()),
            "reliability": float(reliability.item()),
            "coverage": float(coverage.item()),
            "entropy": float(entropy.item()),
            "prototype_coverage": float(prototype_coverage.item()),
            "prior_stability": float(prior_stability.item()),
            "pseudo_peak": float(self.selection_prior.max().item()),
        }
