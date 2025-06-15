# -*- coding: utf-8 -*-
"""model_extraction_attack.py  —  parameter‑tunable version

Adds:
•  --temperature, --n_queries, --student_size CLI flags for the impact‑factor grid.
•  Global QUERY_COUNTER and precise wall‑clock timing for efficiency reporting.
•  query_teacher() helper so every target call is counted.
"""
from __future__ import print_function
import time, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

# 1.  Proprietary target architecture (read‑only for attacker)
from a3_mnist import Lenet as TargetNet

# 2.  Student network (tiny by default; will be patched if --student_size medium)
class StudentNet(nn.Module):
    """A lightweight 2‑conv CNN for MNIST (≈38 k params)."""
    def __init__(self, hidden=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1   = nn.Linear(16 * 7 * 7, hidden)
        self.fc2   = nn.Linear(hidden, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)            # raw logits

# 3.  Data helpers, evaluation metrics

def get_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(test,  batch_size=batch_size, shuffle=False))

@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval(); correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return 100. * correct / total

@torch.no_grad()
def functional_fidelity(student, teacher, loader, device="cpu"):
    student.eval(); teacher.eval(); agree = total = 0
    for x, _ in loader:
        x = x.to(device)
        agree += (student(x).argmax(1) == teacher(x).argmax(1)).sum().item()
        total += x.size(0)
    return 100. * agree / total

# 4.  Global query counter + wrapper so every teacher call is counted
QUERY_COUNTER = 0

def query_teacher(model: nn.Module, x: torch.Tensor, T: float = 5.0):
    global QUERY_COUNTER
    with torch.no_grad():
        QUERY_COUNTER += x.size(0)
        return torch.softmax(model(x) / T, dim=1)

# 5.  Build the soft‑label distillation dataset (uses query_teacher)

def build_extraction_dataset(teacher, pool_loader, device, temperature=5.0):
    teacher.eval(); imgs, soft = [], []
    for bx, _ in pool_loader:
        bx = bx.to(device)
        probs = query_teacher(teacher, bx, temperature)
        imgs.append(bx.cpu()); soft.append(probs.cpu())
    X = torch.cat(imgs)
    Y = torch.cat(soft)
    return TensorDataset(X, Y)

# 6.  Knowledge‑distillation training for the student

def train_student(dataset, val_loader, teacher, device, T=5.0, max_epochs=10):
    loader  = DataLoader(dataset, batch_size=128, shuffle=True)
    student = StudentNet(hidden=256 if args.student_size == 'medium' else 64).to(device)
    opt = optim.Adam(student.parameters(), lr=1e-3)
    kl  = nn.KLDivLoss(reduction='batchmean')

    for ep in range(1, max_epochs + 1):
        student.train(); total = 0.0
        for x, t_logits in loader:
            x, t_logits = x.to(device), t_logits.to(device)
            opt.zero_grad()
            s_logits = student(x)
            loss = kl(F.log_softmax(s_logits / T, dim=1),
                      F.softmax(t_logits / T, dim=1)) * (T*T)
            loss.backward(); opt.step(); total += loss.item()
        print(f"[Student] epoch {ep}  KL={total/len(loader):.4f}")
        if ep % 2 == 0:
            fid = functional_fidelity(student, teacher, val_loader, device)
            print(f"            fidelity={fid:.2f}%")
            if fid >= 97.5: break
    return student

# 7.  CLI, timing, and main orchestration

parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=float, default=5.0)
parser.add_argument('--n_queries',   type=int,   default=60000,
                    help='how many images to query (0=all)')
parser.add_argument('--student_size', choices=['tiny','medium'], default='tiny')
args = parser.parse_args()

def main(device='cpu'):
    t0 = time.perf_counter()

    # 7.1  Load existing target weights or train short demo model
    target = TargetNet().to(device)
    try:
        target.load_state_dict(torch.load('target_model.pth', map_location=device))
        print('[+] Loaded pre-trained target model.')
    except FileNotFoundError:
        print('[!] No target_model.pth found – training 3‑epoch placeholder …')
        target = train_target(device=device)

    # 7.2  Prepare query pool (optionally subsample for --n_queries test)
    train_loader, test_loader = get_dataloaders()
    if args.n_queries and args.n_queries < len(train_loader.dataset):
        idx = random.sample(range(len(train_loader.dataset)), args.n_queries)
        sub_ds = torch.utils.data.Subset(train_loader.dataset, idx)
        train_loader = DataLoader(sub_ds, batch_size=128, shuffle=True)

    # 7.3  Query teacher and build distillation dataset
    distill_ds = build_extraction_dataset(target, train_loader, device, args.temperature)
    v_sz = int(0.10 * len(distill_ds))
    tr_sz = len(distill_ds) - v_sz
    tr_ds, val_ds = torch.utils.data.random_split(distill_ds, [tr_sz, v_sz])
    val_loader = DataLoader(val_ds, batch_size=256)

    # 7.4  Train student
    student = train_student(tr_ds, val_loader, target, device,
                            T=args.temperature)

    # 7.5  Final metrics
    acc_t  = evaluate(target,  test_loader, device)
    acc_s  = evaluate(student, test_loader, device)
    fid    = functional_fidelity(student, target, test_loader, device)

    print('\n==== Extraction Results ====')
    print(f'Target accuracy     : {acc_t:.2f}%')
    print(f'Student accuracy    : {acc_s:.2f}%')
    print(f'Functional fidelity : {fid:.2f}%')

    # 7.6  Efficiency metrics
    wall = time.perf_counter() - t0
    print(f'[Timing] queries={QUERY_COUNTER:,d}  wall‑clock={wall:.1f}s')

    torch.save(student.state_dict(), 'student_model.pth')

if __name__ == '__main__':
    main('cuda' if torch.cuda.is_available() else 'cpu')
