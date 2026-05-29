# Week 2 — The PINN: Architecture, Loss Function & Harmonic Oscillator

> **Goal:** Build a complete, working PINN from scratch on the damped harmonic oscillator.

---

## Topics Covered

- PINN components — the neural network as a surrogate solution `u_θ(x, t)`
- Collocation points — what they replace (the mesh), how to sample them
- Loss function construction:
  - PDE residual loss `L_pde`
  - Initial condition loss `L_ic`
  - Boundary condition loss `L_bc`
  - Total loss `L = L_pde + λ_ic · L_ic + λ_bc · L_bc`
- Soft constraint enforcement — ICs and BCs as weighted penalty terms
- Full worked example — damped harmonic oscillator ODE:  
  `m ẍ + μ ẋ + kx = 0`
- First encounter with spectral bias — what happens when frequency ω₀ is large?

---

## Resources

### 📄 Paper

| Paper | Link |
|---|---|
| Raissi, Perdikaris, Karniadakis — "Physics-Informed Deep Learning (Part I)" J. Comput. Phys. 2019 | [arXiv:1711.10561](https://arxiv.org/abs/1711.10561) |

This is the founding paper of the field. Read it carefully.

### 💻 Notebook

| Resource | Link |
|---|---|
| Ben Moseley — harmonic-oscillator-pinn-workshop **(student version, Colab-ready)** | [GitHub](https://github.com/benmoseley/harmonic-oscillator-pinn-workshop/blob/main/PINN_intro_workshop_student.ipynb) |
| Simpler companion notebook | [GitHub](https://github.com/benmoseley/harmonic-oscillator-pinn) |

> ⚠️ **Open the student version, not the instructor version.** Work through it cell by cell. Do not skip the spectral bias exercise at the end.

### 📝 Blog

| Resource | Link |
|---|---|
| Ben Moseley — "So, what is a physics-informed neural network?" (revisit after completing the notebook) | [benmoseley.blog](https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/) |

### 📖 Review — skim Sections 2–3

| Paper | Link |
|---|---|
| Cuomo et al. — "Scientific Machine Learning Through PINNs: Where we are and What's Next" (2022) | [arXiv:2201.05624](https://arxiv.org/abs/2201.05624) |

---

## Assignment

> Submit a Jupyter notebook + a short written reflection

### Setup

Start from the **completed** harmonic oscillator workshop notebook. The damped harmonic oscillator is:

```
m ẍ + μ ẋ + kx = 0,   x(0) = 1,  ẋ(0) = 0
m = 1,  μ = 0.1
```

The natural frequency is `ω₀ = sqrt(k/m)`.

---

### Task 1 — Frequency Sweep

Train the PINN for each value of `ω₀ ∈ {1, 5, 10, 15, 20}`.

For each `ω₀`, record:
- Final training loss
- Final L² error between predicted `x(t)` and exact solution

Produce **two plots**:
1. Final L² error vs ω₀ (bar chart or line plot)
2. Worst-case: plot predicted vs exact `x(t)` for the ω₀ that performs worst

---

### Task 2 — Written Reflection

Answer these two questions:

1. What is spectral bias in neural networks, and what does your frequency sweep experiment demonstrate about it?
2. Based on what you observed, what would you try to fix the high-frequency failure? (No need to implement it — just hypothesise. We will fix it in Week 5.)

---

## PINN Template

Here is the minimal skeleton every PINN you write this term will follow:

```python
import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, t):
        return self.net(t)

    def loss(self, t_colloc, t_ic):
        # PDE residual
        t_colloc.requires_grad_(True)
        x = self(t_colloc)
        x_t  = torch.autograd.grad(x.sum(), t_colloc, create_graph=True)[0]
        x_tt = torch.autograd.grad(x_t.sum(), t_colloc, create_graph=True)[0]
        pde_residual = x_tt + 0.1 * x_t + (omega**2) * x
        L_pde = (pde_residual**2).mean()

        # Initial conditions
        x0  = self(t_ic)
        x0_t = torch.autograd.grad(x0.sum(), t_ic, create_graph=True)[0]
        L_ic = (x0 - 1.0)**2 + (x0_t - 0.0)**2

        return L_pde + 100 * L_ic
```

---

## Key Concept to Take Away

> A PINN is just a neural network trained to satisfy a differential equation. The PDE is the loss. Observations (ICs, BCs, data) are additional loss terms.

The full loss is:

```
L_total = L_pde + λ_ic * L_ic + λ_bc * L_bc
```

There are no labels. The physics supervises the training.

---

*Next week: We scale up to PDEs - the heat equation and wave equation - and start using the DeepXDE library.*
