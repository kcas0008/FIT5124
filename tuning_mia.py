import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from a3_mnist import Lenet

# 1. Load and prepare the target model
model = Lenet()
model.load_state_dict(torch.load("mnist_cnn.pt", map_location="cpu"))
model.eval()

# 2. Data preparation (same as training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set  = datasets.MNIST('./data', train=False, download=True, transform=transform)

# 3. Feature extraction function for MIA
def get_features(loader):
    features = []
    with torch.no_grad():
        for data, target in loader:
            output = model(data)           # log-prob
            probs = output.exp()           # prob
            entropy = (-probs * probs.log()).sum(dim=1, keepdim=True)
            top2 = probs.topk(2, dim=1).values.cpu().numpy()
            correct = (probs.argmax(1) == target).float().cpu().numpy().reshape(-1, 1)
            feat = np.concatenate([probs.cpu().numpy(), top2, entropy.cpu().numpy(), correct], axis=1)
            features.append(feat)
    return np.vstack(features)

# 4. Prepare data for attack model (limit for speed)
N = 5000
member_features = get_features(DataLoader(torch.utils.data.Subset(train_set, range(N)), batch_size=256))
nonmember_features = get_features(DataLoader(torch.utils.data.Subset(test_set, range(N)), batch_size=256))

X = np.vstack([member_features, nonmember_features])
y = np.hstack([np.ones(len(member_features)), np.zeros(len(nonmember_features))])

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# 5. Function to train and evaluate the attack model
def train_and_evaluate_attack(X_tensor, y_tensor, hidden_layers=[32, 16], lr=1e-3, epochs=100, batch_size=128):
    attack_dataset = TensorDataset(X_tensor, y_tensor)
    attack_loader = DataLoader(attack_dataset, batch_size=batch_size, shuffle=True)
    
    # Build model with variable hidden layers
    layers = []
    prev_dim = X_tensor.shape[1]
    for h in hidden_layers:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(nn.ReLU())
        prev_dim = h
    layers.append(nn.Linear(prev_dim, 1))
    layers.append(nn.Sigmoid())
    attack_model = nn.Sequential(*layers)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=lr)

    for epoch in range(epochs):
        attack_model.train()
        for xb, yb in attack_loader:
            pred = attack_model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    attack_model.eval()
    with torch.no_grad():
        y_prob = attack_model(X_tensor).squeeze().numpy()
        y_pred = (y_prob > 0.5).astype(int)
        acc = accuracy_score(y_tensor.numpy().flatten(), y_pred)
        roc = roc_auc_score(y_tensor.numpy().flatten(), y_prob)
    return acc, roc

# 6. Experiment with different attack model hyperparameters
settings = [
    {"desc": "Default", "hidden_layers": [32, 16], "lr": 1e-3, "epochs": 100},
    {"desc": "Shallow NN", "hidden_layers": [16], "lr": 1e-3, "epochs": 100},
    {"desc": "Deep NN", "hidden_layers": [64, 32, 16], "lr": 1e-3, "epochs": 100},
    {"desc": "Low LR", "hidden_layers": [32, 16], "lr": 1e-4, "epochs": 100},
    {"desc": "High LR", "hidden_layers": [32, 16], "lr": 1e-2, "epochs": 100},
    {"desc": "Few Epochs", "hidden_layers": [32, 16], "lr": 1e-3, "epochs": 20},
    {"desc": "Many Epochs", "hidden_layers": [32, 16], "lr": 1e-3, "epochs": 200},
]

print("Desc\t\tAccuracy\tROC AUC")
for s in settings:
    acc, roc = train_and_evaluate_attack(
        X_tensor, y_tensor,
        hidden_layers=s["hidden_layers"],
        lr=s["lr"],
        epochs=s["epochs"]
    )
    print(f"{s['desc']:<12}\t{acc:.4f}\t\t{roc:.4f}")
