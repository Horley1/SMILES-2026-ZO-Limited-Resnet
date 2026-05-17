"""
zo_optimizer.py — Zero-order optimizer skeleton (student-implemented).

Students: Implement your gradient-free optimization logic inside
``ZeroOrderOptimizer``. The skeleton uses a 2-point central-difference
estimator as a starting point — you are expected to replace or extend it.

Key design points
-----------------
* **Layer selection** is entirely your responsibility. Set ``self.layer_names``
  to the list of parameter names you want to optimize. You can change this list
  at any time — even between ``.step()`` calls — to implement curriculum or
  progressive-layer strategies.
* **Compute budget** is enforced by ``validate.py``: ``.step()`` is called
  exactly ``n_batches`` times. Each call may invoke the model as many times as
  your estimator requires, but be mindful that more evaluations per step leave
  fewer steps in the total budget.
* **No gradients** are computed anywhere in this file. All updates must be
  derived from scalar loss values obtained by calling ``loss_fn()``.

"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    """Gradient-free optimizer for fine-tuning a subset of model parameters.

    The optimizer maintains a list of *active* parameter names
    (``self.layer_names``). On each ``.step()`` call it perturbs only those
    parameters, estimates a pseudo-gradient from forward-pass loss values, and
    applies an update. All other parameters remain strictly frozen.

    Args
    ----
    model             : nn.Module to optimise.
    lr                : Adam learning rate.
    eps               : SPSA perturbation magnitude.
    perturbation_mode : ``"gaussian"`` or ``"uniform"``.
    lora_rank         : Rank of the LoRA.
    lora_alpha        : LoRA scale = alpha / rank.
    n_spsa            : Number of SPSA steps.

    Student task:
        1. Set ``self.layer_names`` to the parameter names you want to tune.
            Inspect available names with ``[n for n, _ in model.named_parameters()]``.
        2. Replace or extend ``_estimate_grad`` with a better estimator.
        3. Replace or extend ``_update_params`` with a better update rule.
        4. Optionally change ``self.layer_names`` inside ``.step()`` to
            implement dynamic layer selection strategies.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.05,
        eps: float = 0.01,
        perturbation_mode: str = "gaussian",
        lora_rank: int = 64,
        lora_alpha  : float = 64.0,
        n_spsa: int = 100,
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps
        self.n_spsa = n_spsa
        self.lora_rank = lora_rank
        self.lora_scale = lora_alpha / lora_rank
        self.step_count = 0

        if perturbation_mode not in ("gaussian", "uniform"):
            raise ValueError(
                f"perturbation_mode must be 'gaussian' or 'uniform', "
                f"got '{perturbation_mode}'"
            )
        self.perturbation_mode = perturbation_mode

        # ------------------------------------------------------------------
        # STUDENT: Set self.layer_names to the parameters you want to tune.
        #
        # The default below selects only the final classification head.
        # You may replace this with any subset of named parameters, e.g.:
        #   self.layer_names = ["layer4.1.conv2.weight", "fc.weight", "fc.bias"]
        #
        # You can also update self.layer_names inside .step() to implement
        # a dynamic schedule (e.g. gradually unfreeze deeper layers).
        # ------------------------------------------------------------------
        self.layer_names: list[str] = [
            "fc.weight",
            "fc.bias",
        ]
        # ------------------------------------------------------------------

        self._lora: dict[str, dict] = {}
        self._direct: dict[str, nn.Parameter] = {}

        self._m: dict[str, torch.Tensor] = {}
        self._v: dict[str, torch.Tensor] = {}

        self._setup()

    # ------------------------------------------------------------------
    # Internal helpers — students may modify these.
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        named = dict(self.model.named_parameters())
        for name in self.layer_names:
            if name not in named:
                continue
            param = named[name]

            if param.dim() >= 2:
                out_dim = param.shape[0]
                in_dim  = param.numel() // out_dim
                r = min(self.lora_rank, out_dim, in_dim)

                B = torch.randn(out_dim, r, device=param.device, dtype=param.dtype) * 0.02
                A = torch.zeros(r, in_dim, device=param.device, dtype=param.dtype)

                self._lora[name] = {
                    "W0":    param.data.reshape(out_dim, in_dim).clone(),
                    "B":     B,   #fixed
                    "A":     A, 
                    "shape": param.shape,
                    "out":   out_dim,
                    "in":    in_dim,
                }
                self._m[f"{name}_A"] = torch.zeros_like(A)
                self._v[f"{name}_A"] = torch.zeros_like(A)
                param.data.copy_(self._effective(name))
            else:
                self._direct[name] = param
                self._m[name] = torch.zeros_like(param)
                self._v[name] = torch.zeros_like(param)

    def _effective(self, name: str) -> torch.Tensor:
        ls = self._lora[name]
        dev = ls["B"].device
        if ls["W0"].device != dev:
            ls["W0"] = ls["W0"].to(dev)
        return (ls["W0"] + self.lora_scale * ls["B"] @ ls["A"]).reshape(ls["shape"])

    def _apply_all(self, params: dict) -> None:
        for name in self._lora:
            if name in params:
                ls = self._lora[name]
                dev = params[name].device
                if ls["B"].device != dev:
                    ls["B"]  = ls["B"].to(dev)
                    ls["A"]  = ls["A"].to(dev)
                    ls["W0"] = ls["W0"].to(dev)
                params[name].data.copy_(self._effective(name))

    def _sample(self, t: torch.Tensor) -> torch.Tensor:
        """Sample a random perturbation vector of the same shape as ``t``."""
        if self.perturbation_mode == "gaussian":
            return torch.randn_like(t)
        return torch.rand_like(t) * 2.0 - 1.0

    def _estimate_grad(
        self, loss_fn: Callable[[], float], params: dict
    ) -> dict[str, torch.Tensor]:
        """Estimate a pseudo-gradient for each active parameter.

        Student task:
            Replace this with a more efficient or accurate estimator.
        """
        # ------------------------------------------------------------------
        # STUDENT: Replace or extend the gradient estimation below.
        # ------------------------------------------------------------------
        acc: dict[str, torch.Tensor] = {}
        for name in self._lora:
            acc[f"{name}_A"] = torch.zeros_like(self._lora[name]["A"])
        for name in self._direct:
            acc[name] = torch.zeros_like(self._direct[name])

        with torch.no_grad():
            for _ in range(self.n_spsa):
                dirs: dict[str, torch.Tensor] = {}
                for name in self._lora:
                    dirs[f"{name}_A"] = self._sample(self._lora[name]["A"])
                for name in self._direct:
                    dirs[name] = self._sample(self._direct[name])

                # +ε
                for name, ls in self._lora.items():
                    ls["A"].add_(self.eps * dirs[f"{name}_A"])
                    params[name].data.copy_(self._effective(name))
                for name, p in self._direct.items():
                    p.data.add_(self.eps * dirs[name])
                f_plus = loss_fn()

                # −ε
                for name, ls in self._lora.items():
                    ls["A"].sub_(2.0 * self.eps * dirs[f"{name}_A"])
                    params[name].data.copy_(self._effective(name))
                for name, p in self._direct.items():
                    p.data.sub_(2.0 * self.eps * dirs[name])
                f_minus = loss_fn()

                # Restore
                for name, ls in self._lora.items():
                    ls["A"].add_(self.eps * dirs[f"{name}_A"])
                    params[name].data.copy_(self._effective(name))
                for name, p in self._direct.items():
                    p.data.add_(self.eps * dirs[name])

                coeff = (f_plus - f_minus) / (2.0 * self.eps)
                for key, u in dirs.items():
                    acc[key].add_(coeff * u)

        return {k: v / self.n_spsa for k, v in acc.items()}
        # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Adam update logic
    # ------------------------------------------------------------------

    def _adam_step(self, key: str, grad: torch.Tensor) -> torch.Tensor:
        b1, b2, eps_a = 0.9, 0.999, 1e-8
        t = self.step_count
        if self._m[key].device != grad.device:
            self._m[key] = self._m[key].to(grad.device)
            self._v[key] = self._v[key].to(grad.device)
        self._m[key] = b1 * self._m[key] + (1.0 - b1) * grad
        self._v[key] = b2 * self._v[key] + (1.0 - b2) * grad * grad
        m_hat = self._m[key] / (1.0 - b1 ** t)
        v_hat = self._v[key] / (1.0 - b2 ** t)
        return self.lr * m_hat / (v_hat.sqrt() + eps_a)

    def _update_params(self, params: dict, grads: dict) -> None:
        """Apply the estimated pseudo-gradients to the active parameters.

        Student task:
            Replace with a more sophisticated update rule, e.g.:
              - Momentum: accumulate an exponential moving average of gradients.
              - Adam-style: maintain first and second moment estimates.
              - Clipped update: ``p ← p - lr * clip(grad, max_norm)``.
        """
        # ------------------------------------------------------------------
        # STUDENT: Replace or extend the parameter update below.
        # ------------------------------------------------------------------
        with torch.no_grad():
            for name, ls in self._lora.items():
                key = f"{name}_A"
                if key in grads:
                    ls["A"].sub_(self._adam_step(key, grads[key]))
                    params[name].data.copy_(self._effective(name))
            for name, p in self._direct.items():
                if name in grads:
                    p.data.sub_(self._adam_step(name, grads[name]))
        # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _active_params(self) -> dict[str, nn.Parameter]:
        """Return a mapping from name → parameter for all active layer names."""
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"The following layer names were not found in the model: "
                f"{missing}. Use [n for n, _ in model.named_parameters()] "
                f"to inspect valid names."
            )
        return {n: named[n] for n in self.layer_names}

    def step(self, loss_fn: Callable[[], float]) -> float:
        """Perform one zero-order optimisation step.

        Calls ``loss_fn`` one or more times to estimate pseudo-gradients for
        the currently active parameters (``self.layer_names``), then applies
        an update. Parameters *not* in ``self.layer_names`` are never touched.

        Args:
            loss_fn: A callable that takes no arguments and returns a scalar
                     ``float`` representing the loss on the current mini-batch.
                     ``validate.py`` guarantees that every call to ``loss_fn``
                     within a single ``.step()`` invocation uses the *same*
                     fixed batch of data.

        Returns:
            The loss value at the *start* of the step (before any update),
            obtained from the first call to ``loss_fn()``.

        Note:
            ``validate.py`` calls ``.step()`` exactly ``n_batches`` times.
            Each forward pass inside ``loss_fn`` counts toward your compute
            budget, so prefer estimators that minimise the number of calls.
        """
        self.step_count += 1
        params = self._active_params()
        self._apply_all(params)

        # Record the loss before any perturbation.
        with torch.no_grad():
            loss_before = loss_fn()

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)
        
        return float(loss_before)