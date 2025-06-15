import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# --- Load model (if running separately, otherwise skip) ---
from a3_mnist import Lenet  # If this code is in another file, uncomment this.
model = Lenet()
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.eval()

# --- Load Data ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set  = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Use a small subset for demonstration
train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256, shuffle=True)

# --- Gather model confidence on train and test data ---
def get_confidences(loader):
    confidences = []
    labels = []
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            probs = output.exp()
            max_conf, _ = probs.max(dim=1)
            confidences.append(max_conf.cpu().numpy())
            labels.append(target.cpu().numpy())
    return np.concatenate(confidences), np.concatenate(labels)

# Get confidence scores
train_conf, train_labels = get_confidences(train_loader)
test_conf,  test_labels  = get_confidences(test_loader)

# --- Membership Inference Attack ---
# The attack: guess "member" if confidence > threshold
all_conf = np.concatenate([train_conf, test_conf])
threshold = np.percentile(all_conf, 95)  # You can tune this

y_true = np.concatenate([np.ones_like(train_conf), np.zeros_like(test_conf)])  # 1 = member, 0 = non-member
y_pred = (np.concatenate([train_conf, test_conf]) > threshold).astype(int)

accuracy = (y_true == y_pred).mean()
print(f"Membership inference attack accuracy: {accuracy:.2f}")

# --- ROC curve (optional, for more analysis) ---
try:
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, np.concatenate([train_conf, test_conf]))
    print(f"Membership inference ROC AUC: {auc:.2f}")
except ImportError:
    pass

# --- Print average confidences ---
print(f"Average train (member) confidence: {train_conf.mean():.4f}")
print(f"Average test  (non-member) confidence: {test_conf.mean():.4f}")
