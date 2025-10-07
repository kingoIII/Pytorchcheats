Here’s a compact, company-ready **PyTorch note sheet**. Minimal fluff, lots of working patterns.

# PyTorch — Working Notes

## 0) Setup & Device

```python
import torch, torch.nn as nn, torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
```

## 1) Tensors (basics)

```python
x = torch.tensor([[1.,2.],[3.,4.]], device=device)   # fixed
y = torch.randn(3, 4, device=device)                 # random ~ N(0,1)
z = torch.zeros_like(y)                              # shape-matched
a = torch.arange(12, device=device).view(3,4)        # reshape
b = a.permute(1,0)                                   # transpose
c = a.mT                                            # matrix transpose
x_np = x.cpu().numpy()                               # to NumPy
```

## 2) Autograd (manual)

```python
w = torch.randn(5, requires_grad=True)
for _ in range(3):
    loss = (w**2).sum()
    loss.backward()
    with torch.no_grad():
        w -= 0.1 * w.grad
        w.grad.zero_()
```

## 3) nn.Module (two patterns)

### A) MLP

```python
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x): return self.net(x)
```

### B) CNN (size-agnostic head)

```python
def conv_bn(in_c, out_c, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, s, p, bias=False),
                         nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

class SmallCNN(nn.Module):
    def __init__(self, in_c=1, num_classes=10, p=0.2):
        super().__init__()
        self.f = nn.Sequential(
            conv_bn(in_c, 32), conv_bn(32, 64), nn.MaxPool2d(2), nn.Dropout2d(p),
            conv_bn(64,128), nn.MaxPool2d(2), nn.Dropout2d(p)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.f(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return self.fc(x)  # logits
```

## 4) Datasets & DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class MyDS(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_loader = DataLoader(MyDS(torch.randn(1000, 20), torch.randint(0,2,(1000,))),
                          batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
```

## 5) Training/Eval loops (clean)

```python
def train_epoch(model, loader, opt, loss_fn):
    model.train(); total=0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total += loss.item()*x.size(0)
    return total/len(loader.dataset)

@torch.no_grad()
def eval_acc(model, loader):
    model.eval(); correct=0; n=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred==y).sum().item(); n += y.size(0)
    return correct/n
```

## 6) Optimizers, Schedulers, AMP

```python
model = SmallCNN(in_c=1, num_classes=10).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
loss_fn = nn.CrossEntropyLoss()

scaler = torch.cuda.amp.GradScaler(enabled=device.type=="cuda")

def train_epoch_amp(model, loader, opt, loss_fn, scaler):
    model.train(); total=0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            logits = model(x)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item()*x.size(0)
    return total/len(loader.dataset)
```

## 7) Saving/Loading

```python
# save
torch.save({"model": model.state_dict(),
            "opt": opt.state_dict()}, "ckpt.pt")

# load
ckpt = torch.load("ckpt.pt", map_location=device)
model.load_state_dict(ckpt["model"])
opt.load_state_dict(ckpt["opt"])
```

## 8) Regularization & Tricks

* **Weight decay** (AdamW): combats overfit on weights.
* **Dropout/Dropout2d**: after activations/blocks.
* **Label smoothing**: `nn.CrossEntropyLoss(label_smoothing=0.1)`
* **Gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`
* **Early stopping**: track valid metric; keep best state_dict.

## 9) Class imbalance

```python
# weights per class (K,)
w = torch.tensor([1.0, 3.0], device=device)
loss_fn = nn.CrossEntropyLoss(weight=w)
```

Or resample with `WeightedRandomSampler`. For harder cases use **Focal Loss**.

## 10) Metrics (quick)

```python
@torch.no_grad()
def topk_acc(model, loader, k=5):
    model.eval(); correct=0; n=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        pred = model(x).topk(k, dim=1).indices
        correct += (pred.eq(y.view(-1,1))).any(dim=1).sum().item()
        n += y.size(0)
    return correct/n
```

## 11) Common losses (selection)

* **Multi-class**: `nn.CrossEntropyLoss` (expects logits, `y∈[0..K-1]`)
* **Binary**: `nn.BCEWithLogitsLoss` (logits vs `{0,1}`), or use CE with 2 classes
* **Regression**: `nn.MSELoss`, `nn.L1Loss`, Huber: `nn.SmoothL1Loss`

## 12) Debugging fast

```python
# Check batch/shape/NaNs
for x,y in train_loader:
    assert x.ndim in (2,4)
    assert not torch.isnan(x).any()
    break

# Sanity overfit on tiny batch
tiny = [next(iter(train_loader))]; model.train()
for _ in range(200):
    x,y = tiny[0][0].to(device), tiny[0][1].to(device)
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), y)
    loss.backward(); opt.step()
# Loss should → ~0 (classification) or very small
```

## 13) Schedulers (useful ones)

```python
# StepLR
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)
# Cosine with warmup (manual warmup example)
for ep in range(epochs):
    if ep < 3:
        for g in opt.param_groups: g['lr'] = 1e-3 * (ep+1)/3
    train_epoch(...)
    sched.step()
```

## 14) Data transforms (vision sketch)

```python
from torchvision import datasets, transforms
mean,std = (0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean,std),
])
test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
test_ds  = datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)
```

## 15) RNN/Seq (minimal)

```python
class TinyRNN(nn.Module):
    def __init__(self, in_dim, hid, num_classes):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hid, batch_first=True)
        self.fc  = nn.Linear(hid, num_classes)
    def forward(self, x):                  # x: (N, T, D)
        h,_ = self.rnn(x)                  # (N, T, H)
        return self.fc(h[:,-1])            # last step → logits
```

## 16) Transformer head (feature → linear)

```python
class LinearHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, feats): return self.fc(feats)   # logits
# Use with frozen backbone outputs
```

## 17) Export (TorchScript / ONNX)

```python
model.eval()
x = torch.randn(1, 1, 28, 28, device=device)
# TorchScript
ts = torch.jit.trace(model, x)
ts.save("model_ts.pt")
# ONNX
torch.onnx.export(model, x, "model.onnx",
                  input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input":{0:"N"}, "logits":{0:"N"}})
```

## 18) Boilerplate — full train loop (classification)

```python
def fit(model, train_loader, val_loader, epochs=10, lr=1e-3, wd=1e-2):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()
    best = {"acc":0.0, "state":None}

    for ep in range(1, epochs+1):
        tr_loss = train_epoch(model, train_loader, opt, loss_fn)
        va_acc  = eval_acc(model, val_loader)
        if va_acc > best["acc"]:
            best = {"acc":va_acc, "state":model.state_dict()}
        print(f"ep {ep:02d} | loss {tr_loss:.4f} | val_acc {va_acc*100:.2f}%")
    model.load_state_dict(best["state"])
    return model
```

## 19) Sanity checklist

* Final layer **out_features = #classes**.
* Use **logits** (no softmax) with `CrossEntropyLoss`.
* BatchNorm **before** ReLU; Dropout **after** activation/blocks.
* Prefer **AdaptiveAvgPool2d(1)** over hard-coded flatten dims.
* `.train()` vs `.eval()` matters (BN/Dropout).
* Verify dataloaders: shuffle train, not test; correct normalization.

## 20) Patterns you’ll reuse

* **Wider first, then deeper** for small data.
* **AdamW + cosine + warmup** is a safe default.
* **Overfit a tiny batch** to confirm the pipeline works.
* **Track LR** and **grad norms** if training is unstable. 
If you want, I can wrap this into a single `notes_pytorch.py` with stubs for MLP/CNN/RNN, training loop, AMP, save/load, and a CLI parser so you can run `--model cnn --dataset cifar10

