# ## Basic tensor operations
import torch

# Create tensors
a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])          # 2x3
b = torch.zeros((2, 3))                                  # 2x3
c = torch.randn(3)                                       # 3
d = torch.arange(6).reshape(2, 3).float()                # 2x3

print("a:", a.shape, a.dtype, a.device)
print("c:", c.shape)

# Reshape/view
flat = a.reshape(-1)                                     # 6
print("flat:", flat.shape)

# Concatenate along rows (dim=0)
cat0 = torch.cat([a, b], dim=0)                          # 4x3
print("cat0:", cat0.shape)

# Broadcasting: add bias vector c (1x3) across rows (2x3)
biased = a + c                                           # broadcast last dim
print("biased:", biased.shape, "expected (2,3)")

# Ensure correct alignment using unsqueeze if needed
bias_col = torch.tensor([10., 20.]).unsqueeze(1)         # 2x1
sum2 = a + bias_col                                      # 2x3 via broadcast
print("sum2:", sum2.shape)


# ## CPU vs GPU
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if hasattr(torch, "mps") and torch.mps.is_available() else device
print("Using device:", device)

def time_matmul(n=4096, iters=10, device="cpu"):
    A = torch.randn(n, n, device=device)
    B = torch.randn(n, n, device=device)
    # Warm-up (especially important for GPU)
    for _ in range(3):
        _ = A @ B
    if device == "cuda":
        torch.cuda.synchronize()
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        C = A @ B

    if device == "cuda":
        torch.cuda.synchronize()
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()

    t1 = time.perf_counter()
    return (t1 - t0) / iters

n = 2048  # adjust down if RAM/GPU limited
cpu_t = time_matmul(n=n, iters=5, device="cpu")
print(f"CPU avg time: {cpu_t:.4f} s")

if device != "cpu":
    gpu_t = time_matmul(n=n, iters=5, device=device)
    print(f"GPU avg time: {gpu_t:.4f} s")
    print("Speedup (CPU/GPU):", cpu_t / gpu_t)
else:
    print("GPU not available; skip GPU timing.")


# ## Autograd mini example
# Data: x and target y_true
x = torch.tensor([0.0, 1.0, 2.0, 3.0]).reshape(-1, 1)         # 4x1
y_true = torch.tensor([1.0, 3.0, 5.0, 7.0]).reshape(-1, 1)    # 4x1

# Parameters with grad tracking
w = torch.randn(1, 1, requires_grad=True)  # 1x1
b = torch.randn(1, requires_grad=True)     # 1

print("w:", w.item())
print("b:", b.item())

# Forward
y_pred = x @ w + b
print("y_pred:", y_pred.reshape(-1))
loss = torch.mean((y_pred - y_true) ** 2)   # MSE
print("Initial loss:", loss.item())

# Backward
loss.backward()
print("w:", w.item(), "w.grad:", w.grad.item())
print("b:", b.item(), "b.grad:", b.grad.item())

# Zero grads for next step
w.grad = None
b.grad = None


# ## Linear regression with autograd (manual SGD) Goal
# Synthetic data: y = 3.0*x + 2.0 + noise
torch.manual_seed(0)
n = 200
X = torch.linspace(-2, 2, n).reshape(-1, 1)                 # n x 1
true_w, true_b = 3.0, 2.0
noise = 0.3 * torch.randn_like(X)
y = true_w * X + true_b + noise

# Parameters
w = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.1
epochs = 200

for epoch in range(1, epochs + 1):
    # Forward
    y_hat = X @ w + b             # n x 1
    loss = torch.mean((y_hat - y) ** 2)

    # Backward
    loss.backward()

    # Parameter update (no_grad to avoid tracking updates)
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    # Zero gradients
    w.grad = None
    b.grad = None

    if epoch % 40 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, w={w.item():.3f}, b={b.item():.3f}")

print(f"True w={true_w}, b={true_b} | Learned w={w.item():.3f}, b={b.item():.3f}")
