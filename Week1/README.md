# Week 1 - PDEs, Neural Network Refresher & Automatic Differentiation

> **Goal:** Understand what a PDE is, refresh your neural network intuition, and learn how to differentiate a neural network with respect to its *inputs* - the one idea that makes PINNs possible.

---

## Topics Covered

- PDE classification - elliptic, parabolic, hyperbolic; reading standard notation (∂u/∂t, ∂²u/∂x²)
- Neural network refresher - MLP architecture, forward pass, loss function, gradient descent
- Automatic differentiation - forward mode vs reverse mode (backpropagation)
- **Key insight:** using `torch.autograd.grad` to differentiate the network output `u(x,t)` with respect to inputs `x` and `t`, not the weights
- PyTorch mechanics - `requires_grad`, `.backward()`, `.grad`, `torch.autograd.grad` for higher-order derivatives

---

## Resources

### Videos

| Resource | Link |
|---|---|
| Steven Brunton - Intro to PDEs playlist (Videos 1–3) | [YouTube](https://www.youtube.com/playlist?list=PLMrJAkhIeNNQromC4WswpU1krLOq5Ro6S) |
| Chris Rackauckas - Intro to Scientific ML 2: PINNs (MIT 18.337) | [YouTube](https://www.youtube.com/watch?v=hKHl68Fdpq4) |
| Karpathy - Neural Networks: Zero to Hero, Lecture 1 (micrograd) | [YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0) |

### Blogs / Reading

| Resource | Link |
|---|---|
| Ben Moseley - "So, what is a physics-informed neural network?" | [benmoseley.blog](https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/) |
| Christopher Olah - "Calculus on Computational Graphs: Backpropagation" | [colah.github.io](https://colah.github.io/posts/2015-08-Backprop/) |

### Coding

| Resource | Link |
|---|---|
| PyTorch Autograd tutorial | [pytorch.org](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) |
| JAX Autodiff Cookbook (read, do not code) | [jax.readthedocs.io](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html) |

### Paper - skim Sections 1-3

| Paper | Link |
|---|---|
| Baydin, Pearlmutter, Radul, Siskind - "Automatic Differentiation in Machine Learning: a Survey" (2018) | [arXiv:1502.05767](https://arxiv.org/abs/1502.05767) |

---


## Assignment

> **Time: ~3h** | Submit a Jupyter notebook

### Tasks

**Task 1 — Build an MLP**  
Implement a 3-layer MLP in PyTorch (no PINN libraries) that takes scalar `x` as input and returns scalar `u(x)`. Use `tanh` activations.

**Task 2 — Compute derivatives using autograd**  
Use `torch.autograd.grad` to compute `du/dx` and `d²u/dx²` for your network.

**Task 3 — Verify**  
Test with `u_true(x) = sin(x)`. Confirm:
- `du/dx ≈ cos(x)`
- `d²u/dx² ≈ −sin(x)`

Evaluate at 100 random points in `[−π, π]`.

**Task 4 — Plot**  
Produce a single figure with 3 subplots:
1. `u(x)` — autograd vs analytical
2. `du/dx` — autograd vs analytical
3. `d²u/dx²` — autograd vs analytical

### Submission

### What we're looking for
- Correct use of `torch.autograd.grad` (not `.backward()`)
- All three plots match the analytical functions closely
- Clean, commented code

---

## Key Concept to Take Away

```python
# This is the core of every PINN:
# differentiating the network OUTPUT w.r.t. the INPUT
x = torch.tensor([1.0], requires_grad=True)
u = network(x)
du_dx = torch.autograd.grad(u, x, create_graph=True)[0]
d2u_dx2 = torch.autograd.grad(du_dx, x)[0]
```

The PDE residual is simply: plug this into your equation and minimise.

---

## Questions to Think About

1. What is the difference between differentiating w.r.t. weights (backprop) vs inputs (for PINNs)?
2. Why does `create_graph=True` matter when computing second-order derivatives?
3. What is spectral bias and why might it matter for solving PDEs? (Hint: look this up briefly — it'll come up in Week 5.)

---

*We'll use these exact tools to build a full PINN on the damped harmonic oscillator.*