import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

from a3_mnist import Lenet

model = Lenet()
model.load_state_dict(torch.load("mnist_cnn.pt", map_location="cpu"))
model.eval()

# 2. Data (same transform as training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set  = datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=256, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=256, shuffle=False)

# 3. Extract features for MIA (full softmax vector, entropy, etc.)
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

# Limit to 5000 train and 5000 test for speed
N = 5000
member_features = get_features(DataLoader(torch.utils.data.Subset(train_set, range(N)), batch_size=256))
nonmember_features = get_features(DataLoader(torch.utils.data.Subset(test_set, range(N)), batch_size=256))

X = np.vstack([member_features, nonmember_features])
y = np.hstack([np.ones(len(member_features)), np.zeros(len(nonmember_features))])

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
attack_dataset = TensorDataset(X_tensor, y_tensor)
attack_loader = DataLoader(attack_dataset, batch_size=128, shuffle=True)

# 4. Attack Model: Simple Neural Net
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

attack_model = AttackNet(X.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(attack_model.parameters(), lr=1e-3)
epochs = 100
for epoch in range(epochs):
    total_loss = 0
    attack_model.train()
    for xb, yb in attack_loader:
        pred = attack_model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    print(f"Attack Epoch {epoch+1}, Loss: {total_loss / len(attack_dataset):.4f}")

# 5. Evaluate Attack Model
attack_model.eval()
with torch.no_grad():
    y_prob = attack_model(X_tensor).squeeze().numpy()
    y_pred = (y_prob > 0.5).astype(int)

print("Membership Attack accuracy:", accuracy_score(y, y_pred))
print("Membership Attack ROC AUC:", roc_auc_score(y, y_prob))