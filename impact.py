import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

# Import LeNet from your own file
from a3_mnist import Lenet

# Optionally, define a simple MLP for more variety:
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# --- Data preparation ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set  = datasets.MNIST('./data', train=False, download=True, transform=transform)

# --- Helper: Train target model ---
def train_target_model(model, train_set, epochs=10, batch_size=128, lr=1e-3):
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    model.train()
    for epoch in range(epochs):
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    return model

# --- Helper: Extract features for MIA ---
def get_features(loader, target_model):
    features = []
    target_model.eval()
    with torch.no_grad():
        for data, target in loader:
            output = target_model(data)
            probs = output.exp()
            entropy = (-probs * probs.log()).sum(dim=1, keepdim=True)
            top2 = probs.topk(2, dim=1).values.cpu().numpy()
            correct = (probs.argmax(1) == target).float().cpu().numpy().reshape(-1, 1)
            feat = np.concatenate([probs.cpu().numpy(), top2, entropy.cpu().numpy(), correct], axis=1)
            features.append(feat)
    return np.vstack(features)

# --- Helper: Train/Evaluate the attack model ---
def train_and_evaluate_attack(X_tensor, y_tensor):
    attack_dataset = TensorDataset(X_tensor, y_tensor)
    attack_loader = DataLoader(attack_dataset, batch_size=128, shuffle=True)
    class AttackNet(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)
    attack_model = AttackNet(X_tensor.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=1e-3)
    epochs = 100
    for epoch in range(epochs):
        attack_model.train()
        for xb, yb in attack_loader:
            pred = attack_model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # Evaluate
    attack_model.eval()
    with torch.no_grad():
        y_prob = attack_model(X_tensor).squeeze().numpy()
        y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_tensor.numpy().flatten(), y_pred)
    roc = roc_auc_score(y_tensor.numpy().flatten(), y_prob)
    return acc, roc

# --- Experiment with different target models/settings ---
target_settings = [
    {"desc": "LeNet 2 epochs (underfit)", "arch": Lenet, "epochs": 2},
    {"desc": "LeNet 10 epochs (standard)", "arch": Lenet, "epochs": 10},
    {"desc": "LeNet 50 epochs (overfit)", "arch": Lenet, "epochs": 50},
    {"desc": "MLP 10 epochs", "arch": SimpleMLP, "epochs": 10}
]

print("Target Model\t\t\tAttack Acc\tROC AUC")
for setting in target_settings:
    print(f"\nTraining target model: {setting['desc']}")
    model = setting["arch"]()
    model = train_target_model(model, train_set, epochs=setting["epochs"])
    # Feature extraction for MIA
    N = 5000
    member_features = get_features(DataLoader(torch.utils.data.Subset(train_set, range(N)), batch_size=256), model)
    nonmember_features = get_features(DataLoader(torch.utils.data.Subset(test_set, range(N)), batch_size=256), model)
    X = np.vstack([member_features, nonmember_features])
    y = np.hstack([np.ones(len(member_features)), np.zeros(len(nonmember_features))])
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    # MIA attack
    acc, roc = train_and_evaluate_attack(X_tensor, y_tensor)
    print(f"{setting['desc']:<30}\t{acc:.4f}\t{roc:.4f}")
