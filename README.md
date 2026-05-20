# Physics-Informed Neural Networks (PINNs)
### WnCC Seasons of Code

---

## What is this project?

Physics-Informed Neural Networks (PINNs) are neural networks that learn to satisfy physical laws (written as differential equations) as part of their training objective. Instead of needing thousands of labelled data points, they learn by minimising how much they violate the governing equations of a system.

This project takes you from building your first PINN on a simple oscillator all the way to applying them to fluid dynamics, inverse problems, or neural operators - depending on the track you choose.

## Prerequisites

You need only these four things before Week 1:

| Requirement | What it means |
|---|---|
| Basic Python | NumPy, loops, matplotlib |
| Calculus | Derivatives, chain rule, reading ∂u/∂x notation |
| Linear algebra | Matrix multiply, dot products (conceptual) |
| Neural networks | Know what an MLP is and that it's trained with gradient descent |


**Optional:**
- [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [Steven Brunton — PDE playlist (first 3 videos)](https://www.youtube.com/playlist?list=PLMrJAkhIeNNQromC4WswpU1krLOq5Ro6S)
- [PyTorch 60-minute blitz](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)
- [Karpathy — micrograd lecture (first 1h)](https://www.youtube.com/watch?v=VMj-3S1tku0)

## Key Papers

| Paper | Why it matters |
|---|---|
| [Raissi, Perdikaris, Karniadakis 2019](https://arxiv.org/abs/1711.10561) | The founding PINN paper |
| [Karniadakis et al. Nature Reviews Physics 2021](https://doi.org/10.1038/s42254-021-00314-5) | The definitive survey |
| [Wang et al. — Expert's Guide 2023](https://arxiv.org/abs/2308.08468) | Best practices for training |
| [Li et al. — FNO 2021](https://arxiv.org/abs/2010.08895) | Fourier Neural Operator |
| [Lu et al. — DeepONet 2021](https://arxiv.org/abs/1910.03193) | Neural operator learning |
