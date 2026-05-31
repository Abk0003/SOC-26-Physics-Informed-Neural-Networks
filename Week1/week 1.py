import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.ln1 = nn.Linear(1,64)
        self.ln2 = nn.Linear(64,64)
        self.ln3 = nn.Linear(64,1)
    def forward(self, x):
        x = F.tanh(self.ln1(x))
        x = F.tanh(self.ln2(x))
        x = F.tanh(self.ln3(x))
        return x

model = NN()
optimizer = torch.optim.SGD(model.parameters())
criterion = nn.MSELoss()

x_train = torch.linspace(-np.pi, np.pi, 1000).unsqueeze(1)
u_train = torch.sin(x_train)

for epoch in range(15000):
    optimizer.zero_grad()
    loss = nn.MSELoss()(model(x_train), u_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:5d} | Loss: {loss.item():.2e}")

def derivative(model, x):
    y = model(x)
    dy_dx = torch.autograd.grad(y,x,create_graph=True,grad_outputs=torch.ones_like(y))[0]
    d2y_dx2 = torch.autograd.grad(dy_dx,x,create_graph=True,grad_outputs=torch.ones_like(dy_dx))[0]
    return y, dy_dx, d2y_dx2

torch.manual_seed(42)
x_np = np.random.uniform(-np.pi, np.pi, 100)
x_np = np.sort(x_np)
x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1).requires_grad_(True)

with torch.no_grad():
    pass

y, dy, d2y = derivative(model, x)

y_true    =  np.sin(x_np)
dy_true   =  np.cos(x_np)
d2y_true  = -np.sin(x_np)

y   = y.detach().numpy()
dy  = dy.detach().numpy()
d2y = d2y.detach().numpy()

print(f"Max |u - sin(x)|        : {np.max(np.abs(y - y_true)):.4f}")
print(f"Max |du/dx - cos(x)|    : {np.max(np.abs(dy - dy_true)):.4f}")
print(f"Max |d²u/dx² + sin(x)|  : {np.max(np.abs(d2y - d2y_true)):.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("MLP Autograd", fontsize=13, fontweight='bold')

titles   = ["u(x)",       "du/dx",      "d²u/dx²"]
a = [y_true,       dy_true,      d2y_true]
nb = [y,    dy,   d2y]
a_labels = ["sin(x)",     "cos(x)",     "−sin(x)"]

for ax, title, ana, net, alabel in zip(axes, titles, a, nb, a_labels):
    ax.plot(x_np, ana, 'k-',  linewidth=2,   label=f"Analytical: {alabel}")
    ax.plot(x_np, net, 'r--', linewidth=1.5, label="Autograd (MLP)")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linewidth=0.5)

plt.tight_layout()
plt.savefig("mlp_autograd.png", dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved → mlp_autograd.png")