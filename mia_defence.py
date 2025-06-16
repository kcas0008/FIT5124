import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from a3_mnist import Lenet

# 1. Load the trained target model
model = Lenet()
model.load_state_dict(torch.load("mnist_cnn.pt", map_location="cpu"))
model.eval()

# 2. Prepare data (same as training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set  = datasets.MNIST('./data', train=False, download=True, transform=transform)

# 3. Evaluate Target Model Accuracy (Defense Accuracy)
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for data, target in DataLoader(test_set, batch_size=256):
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
target_acc = correct / total
print(f"Target Model Test Accuracy: {target_acc:.4f}")

# 4. MIA feature extraction: output restriction (label only)
def get_label_features(loader):
    features = []
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            pred_label = output.argmax(dim=1).cpu().numpy().reshape(-1, 1)
            features.append(pred_label)
    return np.vstack(features)

# Limit for speed
N = 5000
member_features = get_label_features(DataLoader(torch.utils.data.Subset(train_set, range(N)), batch_size=256))
nonmember_features = get_label_features(DataLoader(torch.utils.data.Subset(test_set, range(N)), batch_size=256))

X = np.vstack([member_features, nonmember_features])  # Shape (N*2, 1)
y = np.hstack([np.ones(len(member_features)), np.zeros(len(nonmember_features))])

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
attack_dataset = TensorDataset(X_tensor, y_tensor)
attack_loader = DataLoader(attack_dataset, batch_size=128, shuffle=True)

# 5. Attack Model: Simple neural net for single label input
class AttackNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

attack_model = AttackNet(X.shape[1])
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

# 6. Evaluate the attack model (MIA)
attack_model.eval()
with torch.no_grad():
    y_prob = attack_model(X_tensor).squeeze().numpy()
    y_pred = (y_prob > 0.5).astype(int)

attack_acc = accuracy_score(y, y_pred)
attack_roc = roc_auc_score(y, y_prob)
print("\nMembership Inference Attack with Output Restriction Defense")
print(f"Membership Attack Accuracy: {attack_acc:.4f}")
print(f"Membership Attack ROC AUC: {attack_roc:.4f}")
