# PyTorch 2D Tensors — Fast, Practical Guide
# -----------------------------------------------------------
# Run this file top-to-bottom or piece-by-piece.
# Focus: shape/axes, indexing & slicing, row/col reductions, broadcasting,
# elementwise vs. matrix multiply, and common pitfalls.
#
# Tested with PyTorch >= 1.13

import numpy as np
import pandas as pd
import torch

print("PyTorch:", torch.__version__)

# ---------- 1) Build & Inspect ----------
X = torch.tensor([[11, 12, 13],
                  [21, 22, 23],
                  [31, 32, 33]])
print("\nX:\n", X)
print("ndim:", X.ndim)            # prefer .ndim over legacy .ndimension()
print("shape:", tuple(X.shape))
print("numel:", X.numel())         # total elements

# dtype control
Xf = X.to(torch.float32)
print("dtype X:", X.dtype, "| dtype Xf:", Xf.dtype)

# ---------- 2) NumPy / Pandas interop ----------
# NumPy sharing
np_arr = X.numpy()                 # shares memory for CPU tensors
print("\nnp_arr:\n", np_arr, "| dtype:", np_arr.dtype)

# If we mutate the tensor, NumPy view reflects (and vice versa)
X[0, 0] = -999
print("After edit X[0,0]=-999, np_arr[0,0] ->", np_arr[0,0])
# restore
X[0, 0] = 11

# Pandas DataFrame -> tensor
df = pd.DataFrame({'a':[11,21,31],'b':[12,22,312]})
t_from_df = torch.from_numpy(df.to_numpy())
print("\nFrom DataFrame:\n", t_from_df, "| dtype:", t_from_df.dtype)

# ---------- 3) Indexing & Slicing ----------
print("\nIndexing basics")
print("X[1, 2] ->", X[1, 2], "(row 2, col 3)")
print("X[1][2] ->", X[1][2], "(same, but two steps)")

print("\nRow & column views")
print("Row 0:", X[0])             # 1D view
print("Col 1:", X[:, 1])          # all rows, column 1
print("Last two rows, col 2:", X[1:, 2])

print("\nSlicing gotcha: X[1:3][1] is not 'cols' – it's the second ROW of the slice")
print("X[1:3]:\n", X[1:3])
print("X[1:3][1]:", X[1:3][1])

print("\nHow to pick submatrix by rows & cols: use two steps")
rows = [0, 2]
cols = [1, 2]
sub = X[rows][:, cols]            # first filter rows, then columns
print("Rows [0,2], Cols [1,2]:\n", sub)

print("\nIn-place assignment")
Y = X.clone()
Y[1:, 1] = 0                      # last two rows, second column -> 0
print("Y after Y[1:,1]=0:\n", Y)

# ---------- 4) Reductions along axes ----------
R = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.],
                  [7., 8., 9.]])
print("\nR:\n", R)
print("Column sums (dim=0):", R.sum(dim=0))
print("Row means (dim=1):  ", R.mean(dim=1))
print("Row means keepdim:  ", R.mean(dim=1, keepdim=True))  # column vector shape (3,1)

# ---------- 5) Broadcasting (row/col wise) ----------
# Center each column: subtract column means (shape (3,3) - (3,) broadcasts over rows)
R_centered = R - R.mean(dim=0)
print("\nCentered R by columns:\n", R_centered)

# Add a row bias to every row
row_bias = torch.tensor([10., 20., 30.])         # shape (3,)
R_plus_bias = R + row_bias                        # broadcasts across rows
print("R + row_bias:\n", R_plus_bias)

# If you need explicit 2D shapes for safety:
col_means = R.mean(dim=0, keepdim=True)          # shape (1,3)
R_centered_safe = R - col_means                   # (3,3) - (1,3) -> (3,3)
print("Centered (safe keepdim):\n", R_centered_safe)

# ---------- 6) Elementwise vs Matrix Multiply ----------
A = torch.tensor([[1., 0.],
                  [0., 1.]])
B = torch.tensor([[2., 1.],
                  [1., 2.]])
print("\nElementwise A*B:\n", A * B)             # Hadamard

# Matrix multiplication (shapes (m,n) @ (n,p) -> (m,p))
print("A @ B:\n", A @ B)                         # preferred operator
print("torch.mm(A,B):\n", torch.mm(A, B))

# Matrix-vector: (m,n) @ (n,) -> (m,)
v = torch.tensor([3., 4.])
print("B @ v:", B @ v)

# ---------- 7) Reshape notes ----------
# .view needs contiguous memory; .reshape is often more forgiving
M = torch.arange(12).view(3,4)
print("\nM:\n", M, "| shape:", tuple(M.shape))
print("Flatten then row vector:", M.reshape(-1).shape)  # (12,)
print("As (2, 6):", M.reshape(2, 6).shape)

# ---------- EXERCISES (leave for you) ----------
# E1. Using X defined above, set the second column of the last two rows to 0 in ONE line (without cloning).
#     Verify with print.
#
# E2. Create tensors P (shape 2x3) and Q (shape 3x4) and compute P @ Q.
#     Then try Q @ P and explain the error (or lack thereof).
#
# E3. Given T = torch.tensor([[1.,2.,3.],[4.,5.,6.]]),
#     subtract the column means from T (broadcasting), and compute row sums.
#
# E4. Build a submatrix of R using rows [0,2] and columns [0,2] in one expression.
#     (Hint: two-step indexing like X[rows][:, cols])
#
# E5. For K = torch.arange(16).view(4,4):
#     a) grab the last column with a 2D shape (4,1) (hint: keepdim or slicing with None/unsqueeze)
#     b) reverse the rows (hint: slicing with step -1)
# -----------------------------------------------------------
