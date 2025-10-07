You want a compact, no-nonsense CNN notes pack. Here it is.

# CNN — Minimal but Complete Notes (PyTorch)

## Core ideas (keep handy)

* Convolution: slide kernel over input, share weights, extract local patterns.
* Output size (each dim):
  [
  \text{out}=\left\lfloor\frac{\text{in}+2p-k}{s}\right\rfloor+1
  ]
  where (k)=kernel, (s)=stride, (p)=padding.
* Ordering (common): Conv → **BatchNorm** → ReLU → (Dropout?) → Pool.
* BatchNorm: stabilize/accelerate training; use **before** activation.
* Dropout: regularize; often after dense layers, sometimes after conv blocks.
* Flatten trap: don’t hard-code shape; use **AdaptiveAvgPool** or infer dynamically.
* Global Average Pool (GAP): reduces (C,H,W) → (C); size-agnostic.
* Classification head size = **#classes** (e.g., 10 for MNIST/CIFAR-10).
* Optim: Adam/SGD; schedule helps (StepLR/Cosine).
* Data aug: essential for CIFAR-10; optional for MNIST.

---

## One-file template you can reuse

* Supports MNIST or CIFAR-10
* Toggles: BatchNorm, Dropout, Data Aug, GlobalAvgPool
* Size-agnostic; no magic `49`

```python
# cnn_notes.py
import argparse, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -------------------------
# Model blocks
# -------------------------
def conv_block(in_c, out_c, k=3, s=1, p=1, use_bn=True, p_drop=0.0):
    layers = [nn.Conv2d(in_c, out_c, k, s, p, bias=not use_bn)]
    if use_bn:
        layers += [nn.BatchNorm2d(out_c)]
    layers += [nn.ReLU(inplace=True)]
    if p_drop > 0:
        layers += [nn.Dropout2d(p_drop)]
    return nn.Sequential(*layers)

class SimpleCNN(nn.Module):
    """
    Minimal CNN: 2 conv blocks + GAP + Linear.
    Size-agnostic: works for 28x28 (MNIST) or 32x32 (CIFAR10) without shape hacks.
    """
    def __init__(self, in_channels=1, num_classes=10, use_bn=True, p_drop=0.0, hidden_mult=1.0, use_gap=True):
        super().__init__()
        c1, c2 = int(16*hidden_mult), int(32*hidden_mult)
        self.features = nn.Sequential(
            conv_block(in_channels, c1, use_bn=use_bn, p_drop=p_drop),
            nn.MaxPool2d(2),                 # downsample x2
            conv_block(c1, c2, use_bn=use_bn, p_drop=p_drop),
            nn.MaxPool2d(2),                 # downsample x2
        )
        self.use_gap = use_gap
        if use_gap:
            self.head_gap = nn.AdaptiveAvgPool2d(1)  # -> (N, C, 1,1)
            self.fc = nn.Linear(c2, num_classes)
        else:
            # Infer flatten dim dynamically
            self._nflat = None
            self.fc = nn.LazyLinear(num_classes)
        # optional final dropout (dense)
        self.dropout_fc = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.features(x)
        if self.use_gap:
            x = self.head_gap(x).squeeze(-1).squeeze(-1)  # (N, C)
        else:
            x = torch.flatten(x, 1)                       # (N, C*H*W)
        x = self.dropout_fc(x)
        return self.fc(x)

# -------------------------
# Data
# -------------------------
def get_data(dataset="mnist", batch_size=128, aug=False):
    if dataset.lower() == "mnist":
        tf_train = [transforms.ToTensor()]
        tf_test  = [transforms.ToTensor()]
        train = datasets.MNIST("./data", train=True, download=True, transform=transforms.Compose(tf_train))
        test  = datasets.MNIST("./data", train=False, download=True, transform=transforms.Compose(tf_test))
        in_ch = 1; num_classes = 10
    elif dataset.lower() == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        tf_train = []
        if aug:
            tf_train += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        tf_train += [transforms.ToTensor(), transforms.Normalize(mean, std)]
        tf_test   = [transforms.ToTensor(), transforms.Normalize(mean, std)]
        train = datasets.CIFAR10("./data", train=True, download=True, transform=transforms.Compose(tf_train))
        test  = datasets.CIFAR10("./data", train=False, download=True, transform=transforms.Compose(tf_test))
        in_ch = 3; num_classes = 10
    else:
        raise ValueError("dataset must be mnist or cifar10")
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
            DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
            in_ch, num_classes)

# -------------------------
# Train / Eval
# -------------------------
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred==y).sum().item()
            total += y.size(0)
    return correct/total

def train(model, train_loader, test_loader, epochs=5, lr=1e-3, device="cuda", use_amp=True, step_lr=None):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_lr, gamma=0.5) if step_lr else None
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.startswith("cuda"))
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        model.train()
        t0, running = time.time(), 0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()*x.size(0)

        if sched: sched.step()
        train_loss = running/len(train_loader.dataset)
        test_acc = accuracy(model, test_loader, device)
        print(f"Epoch {ep:02d} | loss {train_loss:.4f} | test_acc {test_acc*100:.2f}% | lr {opt.param_groups[0]['lr']:.5f} | {time.time()-t0:.1f}s")

    return model

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="mnist", choices=["mnist","cifar10"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bn", action="store_true", help="use BatchNorm (on by default in conv_block); kept for explicitness")
    ap.add_argument("--no_bn", action="store_true", help="disable BatchNorm in conv blocks")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--aug", action="store_true", help="enable train-time augmentation (CIFAR10 recommended)")
    ap.add_argument("--hidden_mult", type=float, default=1.0, help="scale channels width")
    ap.add_argument("--no_gap", action="store_true", help="disable GAP and use flatten")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--epochs_step_lr", type=int, default=0, help="if >0, StepLR step size (epochs)")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    train_loader, test_loader, in_ch, num_classes = get_data(args.dataset, args.batch_size, args.aug)
    use_bn = not args.no_bn
    use_gap = not args.no_gap

    model = SimpleCNN(in_channels=in_ch,
                      num_classes=num_classes,
                      use_bn=use_bn,
                      p_drop=args.dropout,
                      hidden_mult=args.hidden_mult,
                      use_gap=use_gap)

    train(model, train_loader, test_loader,
          epochs=args.epochs,
          lr=args.lr,
          device=device,
          use_amp=(device=="cuda"),
          step_lr=(args.epochs_step_lr if args.epochs_step_lr>0 else None))

if __name__ == "__main__":
    main()
```

### Quick runs

```bash
# MNIST baseline (no aug needed)
python cnn_notes.py --dataset mnist --epochs 3

# MNIST with BN + Dropout
python cnn_notes.py --dataset mnist --epochs 5 --dropout 0.3

# CIFAR-10 with augmentation (recommended)
python cnn_notes.py --dataset cifar10 --epochs 25 --aug --dropout 0.2 --hidden_mult 1.5 --epochs_step_lr 8
```

---

## Patterns to memorize

* **BN placement:** `Conv → BN → ReLU`.
* **Dropout placement:** after activation; heavier in fully-connected head.
* **Pooling:** use MaxPool(2) or Strided Conv for downsampling.
* **GAP vs Flatten:** Prefer **GAP** for fewer params + robustness.
* **Init:** PyTorch defaults are fine; for deeper nets consider Kaiming (He) init.
* **Normalization (images):**

  * MNIST: typically just `ToTensor()`.
  * CIFAR-10: normalize with dataset mean/std.

---

## Common mistakes (and how to not be that person)

* Hard-coding flatten dim (e.g., `out_2*49`). Use GAP or LazyLinear.
* Wrong num_classes in final Linear (must match labels).
* Forgetting `.train()` / `.eval()` modes (BN/Dropout behave differently).
* Learning rate too high with BN → exploding/oscillating loss.
* Batch size of 1 with BN → noisy stats; prefer ≥32 if possible.
* Misplaced BN (after ReLU) reduces effect.

---

## Extend when ready

* **Deeper stacks:** add `conv_block`s: 16→32→64→128 channels.
* **Strided conv:** replace MaxPool with `Conv2d(..., stride=2)`.
* **Cosine LR:** try `torch.optim.lr_scheduler.CosineAnnealingLR`.
* **Label smoothing:** `CrossEntropyLoss(label_smoothing=0.1)` (PyTorch ≥1.10).
* **MixUp/CutMix** for CIFAR-10 if you want stronger regularization.

---

## Tiny checklist for exams/quizzes

* Output size formula ✔
* BN purpose & placement ✔
* Dropout purpose & typical placement ✔
* GAP vs Flatten ✔
* Final layer = #classes ✔
* MNIST vs CIFAR-10 differences ✔
* Don’t hard-code shapes ✔

That’s your portable CNN notes + runnable code. Use it, modify it, and stop guessing shapes.
