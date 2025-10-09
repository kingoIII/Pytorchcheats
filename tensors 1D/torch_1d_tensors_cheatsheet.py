# PyTorch 1D Tensors — Compact Cheat Sheet
# -----------------------------------------------------------
# Run this file section-by-section to explore 1D tensors.
# Tested with PyTorch >= 1.13. Key ideas:
#  - dtype vs. type(), shape via .size() / .ndim
#  - reshape with .view() (elements count must stay constant; only ONE -1 allowed)
#  - NumPy <-> torch memory sharing
#  - indexing/slicing/advanced indexing
#  - common reductions: mean/std/min/max
#  - elementwise ops, broadcasting, dot product
#  - torch.linspace + torch.sin for quick function plots

import math
import numpy as np

try:
    import torch
    import matplotlib.pyplot as plt
except Exception as e:
    print("Imports failed here. Make sure you have torch and matplotlib installed in your environment.")
    raise

print("PyTorch version:", torch.__version__)

# ---------- Types & Shape ----------
ints = torch.tensor([0, 1, 2, 3, 4])                     # Long by default for integers
floats = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])         # Float32 by default for floats

print("\n[dtype] ints:", ints.dtype, "| floats:", floats.dtype)
print("[type()] ints:", ints.type(), "| floats:", floats.type())
print("[ndim,size] ints:", ints.ndim, ints.size())

# Convert dtype explicitly
as_float = ints.to(torch.float32)
as_long  = floats.to(torch.int64)
print("\n[cast] ints->float32:", as_float.dtype, "| floats->int64:", as_long.dtype)

# Reshape with view (same #elements; only ONE -1 allowed)
v = torch.arange(5)
col = v.view(5, 1)     # 5x1
same_col = v.view(-1, 1)
print("\n[view] v.shape:", v.shape, "| col.shape:", col.shape, "| same_col.shape:", same_col.shape)

# ---------- NumPy Interop (memory sharing) ----------
np_arr = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
t_from_np = torch.from_numpy(np_arr)  # shares memory
print("\n[np->torch] dtype:", t_from_np.dtype, "| type:", t_from_np.type())
np_back = t_from_np.numpy()           # still the same underlying buffer

np_arr[:] = 42.0                      # mutate NumPy -> reflects in both views
print("[sharing] t_from_np:", t_from_np, "| np_back:", np_back)

# ---------- Indexing & Slicing ----------
x = torch.tensor([100, 1, 2, 300, 400])
print("\n[indexing] x[0]:", x[0], "| x[-1]:", x[-1])
print("[slicing] x[1:4]:", x[1:4])

# Advanced indexing with a list of positions returns a copy
idx = [1, 3]
print("[adv idx] x[idx]:", x[idx])

# In-place assign with slicing / advanced indexing
x[3:5] = torch.tensor([30, 40])
x[idx] = 99999
print("[assign] x after edits:", x)

# .item() only on a 0-dim (scalar) tensor or a single-element tensor
scalar = torch.tensor(7)
print("\n[item] scalar.item():", scalar.item())

# Convert to Python list
lst = torch.tensor([0, 1, 2, 3]).tolist()
print("[tolist] ->", lst)

# ---------- Reductions / Functions ----------
m = torch.tensor([1.0, -1.0, 1.0, -1.0])
print("\n[mean,std] mean:", m.mean().item(), "| std:", m.std(unbiased=True).item())

mm = torch.tensor([1, 1, 3, 5, 5])
print("[min,max] min:", mm.min().item(), "| max:", mm.max().item())

# ---------- Elementwise Ops & Dot ----------
u = torch.tensor([1, 2], dtype=torch.float32)
v = torch.tensor([3, 2], dtype=torch.float32)

print("\n[elementwise] u+v:", (u+v))
print("[elementwise] u*v:", (u*v))
print("[scalar add] u+1:", (u+1))
print("[scalar mul] 2*u:", (2*u))

# Dot product: 1*3 + 2*2 = 7
print("[dot] torch.dot(u,v):", torch.dot(u, v).item())

# ---------- linspace & sin ----------
t = torch.linspace(0, 2*math.pi, steps=100)
s = torch.sin(t)

# Plot (single chart; no explicit colors to keep defaults)
plt.figure()
plt.plot(t.numpy(), s.numpy())
plt.title("sin(x) from 0 to 2π")
plt.xlabel("x (radians)")
plt.ylabel("sin(x)")
plt.grid(True)
plt.show()

# ---------- YOUR TURN (leave as exercises) ----------
# 1) Make a tensor with 25 steps in [0, π/2], compute sin, print min/max, and plot it.
#    Hint: t25 = torch.linspace(0, 0.5*math.pi, steps=25)
#    Then: s25 = torch.sin(t25)
#    Print: s25.min().item(), s25.max().item()
#    Plot similarly to the example above.
#
# 2) Given practice_tensor = torch.arange(10).clone(), set indices [3,4,7] to 0 using advanced indexing.
#    Verify with print.
#
# 3) Given a = torch.arange(12), which of these will fail and why?
#       a.view(3, 4)     a.view(2, 2, 3)     a.view(-1, 4)     a.view(-1, -1)
#    Explain briefly.
# -----------------------------------------------------------
