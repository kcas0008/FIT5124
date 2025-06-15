import time, random, hashlib, collections, statistics, json, pathlib
from typing import Optional, Dict, List

import torch
import torch.nn.functional as F

# 1) QUERY-COST & IN-FLIGHT DETECTION

class QueryMonitor:
    def __init__(self,
                 n_max: int = 60_000,      # allow up to MNIST train size / client
                 window: int = 2_000,      # run χ² every 2k queries
                 chi2_thresh: float = 15.0 # p≈0.05 for 9 dof
                 ):
        self.n_max     = n_max
        self.window    = window
        self.chi2_th   = chi2_thresh
        self.per_ip    : Dict[str, int] = collections.defaultdict(int)
        self.histogram : Dict[int, int] = collections.defaultdict(int)
        self.start     = time.time()
        pathlib.Path('logs').mkdir(exist_ok=True)

    # --------------------------------------------------------------------- #
    def _log(self, client_ip: str, preds: torch.Tensor):
        ts  = time.time() - self.start
        top = preds.argmax(1).tolist()
        for t in top: self.histogram[t] += 1
        self.per_ip[client_ip] += len(preds)

        entry = {'t': ts,
                 'ip': client_ip,
                 'n':  len(preds),
                 'top_classes': top}
        with open('logs/query_log.jsonl', 'a') as f:
            f.write(json.dumps(entry) + '\n')

        # χ² scan
        total = sum(self.histogram.values())
        if total % self.window == 0:
            exp = total / 10.0
            chi2 = sum((c - exp) ** 2 / exp for c in self.histogram.values())
            if chi2 < self.chi2_th:
                print(f'[!] χ²={chi2:.2f} – POSSIBLE UNIFORM SCANNING ATTACK')

    # --------------------------------------------------------------------- #
    def register(self, client_ip: str, preds: torch.Tensor):
        n_so_far = self.per_ip[client_ip]
        if n_so_far >= self.n_max:
            raise RuntimeError(f'Client {client_ip} exceeded query budget '
                               f'({self.n_max}).')
        self._log(client_ip, preds)

# 2) PROBABILISTIC DEFENCE WRAPPER  (+ adaptive hardening)
class ExtractionDefenceWrapper(torch.nn.Module):
    def __init__(self,
                 model: torch.nn.Module,
                 monitor: QueryMonitor,
                 *,
                 base_sigma: float = 0.10,       # σ when client has 0 queries
                 sigma_growth: float = 0.75,     # extra σ at full budget
                 round_ndec: int  = 0,
                 top_k: int       = 1,
                 base_flip: float = 0.05,        # p_flip at 0 queries
                 flip_growth: float = 0.45,      # additional flip at full budget
                 return_log: bool = False):
        super().__init__()
        self.model         = model
        self.monitor       = monitor
        self.base_sigma    = base_sigma
        self.sigma_growth  = sigma_growth
        self.round_ndec    = round_ndec
        self.top_k         = top_k
        self.base_flip     = base_flip
        self.flip_growth   = flip_growth
        self.return_log    = return_log

    # --------------------------------------------------------------------- #
    def _noise_and_flip_params(self, client_ip: str) -> (float, float):
        """Linear schedule based on fraction of budget already used."""
        frac = min(1.0, self.monitor.per_ip[client_ip] /
                          max(1, self.monitor.n_max))
        sigma = self.base_sigma + frac * self.sigma_growth
        pflip = self.base_flip + frac * self.flip_growth
        return sigma, pflip

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def forward(self, x: torch.Tensor, client_ip: str = 'anonymous'):
        logits = self.model(x)

        # 1️⃣  Dynamic noise & dummy-label params
        sigma, pflip = self._noise_and_flip_params(client_ip)
        if sigma > 0:
            logits = logits + torch.randn_like(logits) * sigma

        probs = F.softmax(logits, dim=1)

        # 2️⃣  Dummy-label injection
        if pflip > 0:
            top2_val, top2_idx = probs.topk(2, dim=1)
            flip_mask = (torch.rand(x.size(0), device=x.device) < pflip)
            probs[flip_mask] = 0
            # random wrong label (stronger than 2nd-best)
            rand_lab = torch.randint(0, 10, (flip_mask.sum().item(),),
                                     device=x.device)
            probs[flip_mask, rand_lab] = 1.0
            probs[~flip_mask] = 0
            probs[~flip_mask, top2_idx[~flip_mask, 0]] = 1.0
        else:
            top_vals, top_idx = probs.topk(self.top_k, dim=1)
            out = torch.zeros_like(probs)
            out.scatter_(1, top_idx, top_vals)
            probs = out

        # 3️⃣  Coarse rounding
        if self.round_ndec is not None:
            factor = 10 ** self.round_ndec
            probs = torch.round(probs * factor) / factor

        # 4️⃣  Monitoring / rate limiting
        self.monitor.register(client_ip, probs)

        return probs.log() if self.return_log else probs

# 3) WATERMARK SET  –  post-hoc theft detection
class Watermark:
    def __init__(self,
                 seed_loader: torch.utils.data.DataLoader,
                 teacher: torch.nn.Module,
                 n_canary: int = 128,
                 tau: float    = 0.85,      # 85 % agreement ⇒ likely stolen
                 device: str   = 'cpu'):
        self.device = device
        self.tau    = tau
        self.inputs, self.labels = self._build(seed_loader, teacher, n_canary)

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def _build(self, loader, teacher, n) -> (torch.Tensor, torch.Tensor):
        data = []
        # 1) sample random inputs, but *keep only those where the teacher
        #    is extremely confident* (max prob > 0.995) – improves robustness
        for x, _ in loader:
            x = x.to(self.device)
            logits = teacher(x)
            p, y = logits.softmax(1).max(1)
            mask = p > 0.995
            if mask.any():
                data.append((x[mask].cpu(), y[mask].cpu()))
            if sum(t.size(0) for t,_ in data) >= n:
                break
        xs = torch.cat([t for t,_ in data])[:n]
        ys = torch.cat([l for _,l in data])[:n]
        return xs, ys

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def check_suspect_model(self, suspect: torch.nn.Module) -> bool:
        suspect.eval()
        outs = suspect(self.inputs.to(self.device)).argmax(1).cpu()
        agree = (outs == self.labels).float().mean().item()
        print(f'[Watermark] agreement = {agree*100:.1f}% '
              f"(threshold {self.tau*100:.0f}%)")
        return agree >= self.tau
