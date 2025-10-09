
<img width="1746" height="1101" alt="output" src="https://github.com/user-attachments/assets/0ef06e97-7c2c-4a1a-9511-b69d2628b9c2" />
What it covers (tight and practical):

* dtype vs. type(), shape/ndim, and safe casting with `.to(...)`
* Reshaping with `.view(...)` (why only **one** `-1` is allowed)
* NumPy ↔️ torch memory sharing (mutate NumPy, see tensor change)
* Indexing, slicing, advanced indexing + in-place assignment
* Reductions: `.mean()`, `.std()`, `.min()`, `.max()`
* Elementwise ops, broadcasting, and `torch.dot`
* `torch.linspace` + `torch.sin` with a quick plot
* 
To keep the learning active, your cheat sheet ends with three short “YOUR TURN” tasks (commented). Do them in your editor:

1. Build a 25-step tensor on `[0, π/2]`, take `sin`, print min/max, plot.
2. With `practice_tensor = torch.arange(10)`, set indices `[3, 4, 7]` to `0`.
3. Given `a = torch.arange(12)`, which reshape fails:
   `a.view(3, 4)`, `a.view(2, 2, 3)`, `a.view(-1, 4)`, `a.view(-1, -1)` — and *why*?

One quick check to be sure the core idea stuck:
If `x = torch.arange(6)`, will `x.view(2, -1)` work, and what shape will it be? (Answer it in your own words, then we’ll move on to broadcasting.)
