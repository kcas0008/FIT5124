# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

# 1.  Import the *proprietary* target architecture (Lenet).
from a3_mnist import Lenet as TargetNet  # vendor model (read‑only)

# 2.  Definition of the *student* network (≈ 38 k parameters).
#     Much smaller than the vendor model – proves we can replicate behaviour
#     without knowing/duplicating its capacity.
class StudentNet(nn.Module):
    """A lightweight 2‑conv CNN for MNIST (≈ 38 k params)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # raw logits

# 3.  Utility functions: dataloaders, evaluation, fidelity, etc.

def get_dataloaders(batch_size=128):
    """Standard MNIST loaders with mean/var normalisation."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test = datasets.MNIST("./data", train=False, download=True, transform=transform)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size, shuffle=False),
    )


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    """Compute ground‑truth accuracy (%) on *loader*."""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def functional_fidelity(student, target, loader, device="cpu"):
    """Agreement (% of identical top‑1 predictions) between *student* and *target*."""
    student.eval(); target.eval()
    agree, total = 0, 0
    for x, _ in loader:
        x = x.to(device)
        agree += (student(x).argmax(1) == target(x).argmax(1)).sum().item()
        total += x.size(0)
    return 100.0 * agree / total

# 4.  (Optional) quick training routine for the *target* if weights absent.
#     In real life the attacker **would not** have this much access, but we
#     include the code so the script is fully reproducible offline.

def train_target(epochs=3, device="cpu"):
    target = TargetNet().to(device)
    opt = optim.Adadelta(target.parameters(), lr=1.0)
    train_loader, test_loader = get_dataloaders()
    target.train()
    for ep in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.nll_loss(target(x), y)
            loss.backward(); opt.step()
        print(f"[Target] epoch {ep+1}/{epochs} finished …")
    torch.save(target.state_dict(), "target_model.pth")
    acc = evaluate(target, test_loader, device)
    print(f"[Target] test accuracy: {acc:.2f}%")
    return target


# ---------------------------------------------------------------------------
# 5.  Build the synthetic distillation dataset by querying the target model.
# ---------------------------------------------------------------------------

def build_extraction_dataset(
        target: torch.nn.Module,
        pool_loader: torch.utils.data.DataLoader,
        device: torch.device,
        temperature: float = 5.0
) -> torch.utils.data.TensorDataset:
    target.eval()                           # just to be explicit
    imgs, soft_labels = [], []

    with torch.no_grad():                   # <-- key: no autograd graph!
        for batch_imgs, _ in pool_loader:   # second element is a dummy label
            batch_imgs = batch_imgs.to(device)

            # Forward pass through the teacher
            logits = target(batch_imgs)                 # shape (B, 10)
            probs  = torch.softmax(logits / temperature, dim=1)

            # Move everything to CPU tensors detached from graph
            imgs.append(batch_imgs.cpu())
            soft_labels.append(probs.cpu())

    # Concatenate into single tensors
    X = torch.cat(imgs, dim=0)             # shape (N, 1, 28, 28)
    Y = torch.cat(soft_labels, dim=0)       # shape (N, 10)

    return torch.utils.data.TensorDataset(X, Y)



# ---------------------------------------------------------------------------
# 6.  Train the *student* via temperature‑scaled knowledge‑distillation.
# ---------------------------------------------------------------------------

def train_student(dataset, val_loader, target, device="cpu", T=5.0, epochs=10):
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    student = StudentNet().to(device)
    opt = optim.Adam(student.parameters(), lr=1e-3)
    kl = nn.KLDivLoss(reduction="batchmean")

    for ep in range(epochs):
        student.train(); total_loss = 0.0
        for x, tgt_logits in loader:
            x, tgt_logits = x.to(device), tgt_logits.to(device)
            opt.zero_grad()
            s_logits = student(x)
            loss = kl(
                F.log_softmax(s_logits / T, dim=1),
                F.softmax(tgt_logits / T, dim=1)
            ) * (T * T)  # scale back distillation loss
            loss.backward(); opt.step(); total_loss += loss.item()
        print(f"[Student] epoch {ep+1}/{epochs}   KL={total_loss / len(loader):.4f}")

        # quick fidelity check on validation split every 2 epochs
        if ep % 2 == 1:
            fid = functional_fidelity(student, target, val_loader, device)
            print(f"           fidelity={fid:.2f}%")
            if fid > 97.5:  # early stop once we surpass 97.5 % agreement
                break

    return student

# 7.  Main routine tying everything together.
def main(device="cpu"):
    # 7.1  Load or (if absent) train the target model
    try:
        target = TargetNet().to(device)
        target.load_state_dict(torch.load("target_model.pth", map_location=device))
        print("[+] Loaded pre‑trained target model.")
    except FileNotFoundError:
        print("[!] target_model.pth not found – training a fresh target (3 epochs)…")
        target = train_target(device=device)

    # 7.2  Create the extraction dataset
    train_loader, test_loader = get_dataloaders()
    distill_ds = build_extraction_dataset(target, train_loader, device)

    # 7.3  90/10 split gives us a small validation set for early‑stopping
    val_sz = int(0.10 * len(distill_ds))
    train_sz = len(distill_ds) - val_sz
    tr_ds, val_ds = torch.utils.data.random_split(distill_ds, [train_sz, val_sz])
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    # 7.4  Train the student
    student = train_student(tr_ds, val_loader, target, device)

    # 7.5  Final evaluation
    acc_tgt = evaluate(target, test_loader, device)
    acc_std = evaluate(student, test_loader, device)
    fid = functional_fidelity(student, target, test_loader, device)

    print("\n==== Extraction Results ====")
    print(f"Target accuracy       : {acc_tgt:.2f}%")
    print(f"Student accuracy      : {acc_std:.2f}%")
    print(f"Functional fidelity   : {fid:.2f}% (agreement with oracle)")

    # Save stolen weights for later misuse
    torch.save(student.state_dict(), "student_model.pth")

if __name__ == "__main__":
    main("cuda" if torch.cuda.is_available() else "cpu")
