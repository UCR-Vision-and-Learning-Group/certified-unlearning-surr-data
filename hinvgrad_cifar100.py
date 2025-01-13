import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.autograd import grad
from tqdm import tqdm

from src.loss import L2RegularizedCrossEntropyLoss
from src.data import get_train_test_datasets, get_transforms
from torchvision.models import resnet18, ResNet18_Weights, alexnet, AlexNet_Weights
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import InterpolationMode
import logging
import time

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='hinvgrad_cifar100.log',
    filemode='w'
)


import torch
import torch.nn as nn
import torch.nn.functional as F

# class CNNModel(nn.Module):
#     def __init__(self, num_classes):
#         super(CNNModel, self).__init__()
#         # Convolutional Block 1
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(0.25)
#
#         # Convolutional Block 2
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#
#         # Fully Connected Block
#         self.fc1 = nn.Linear(128*8*8, 512)  # Assuming 32x32 input
#         self.fc2 = nn.Linear(512, num_classes)
#         self.dropout_fc = nn.Dropout(0.5)
#
#     def forward(self, x):
#         # Block 1
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = self.dropout(x)
#
#         # Block 2
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = self.pool(x)
#         x = self.dropout(x)
#
#         # Fully Connected
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc1(x))
#         x = self.dropout_fc(x)
#         x = self.fc2(x)
#
#         return x

# Define LeNet-5 network
# class LeNet5(nn.Module):
#     def __init__(self, num_classes=10):
#         super(LeNet5, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
#         self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.max_pool2d(x, 2)
#         x = torch.relu(self.conv2(x))
#         x = torch.max_pool2d(x, 2)
#         x = x.view(x.size(0), -1)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowCNN128(nn.Module):
    def __init__(self, num_classes):
        super(ShallowCNN128, self).__init__()
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # Adjust for 128x128 input size
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Forward pass through convolutional blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # Flatten and pass through fully connected layers
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Function to evaluate a model
def evaluate_model(model, dataloader, criterion, device, name="Dataset", log=True):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in tqdm(dataloader, desc=f"Evaluating on {name}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch, model)
            total_loss += loss.item() * X_batch.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total * 100
    print(f"Evaluation on {name}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
    if log:
        logging.info(f"Evaluation on {name}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
    return avg_loss, accuracy


# Function to compute batched Hessian-vector product (HVP)
def batched_hvp(model, dataloader, criterion, params, vector, device):
    hvp_result = None
    total_item = 0
    for X_batch, y_batch in tqdm(dataloader, desc="Computing HVP"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch, model)

        gradients = grad(loss, params, create_graph=True)
        grad_vector = torch.cat([g.view(-1) for g in gradients if g is not None])
        hvp_batch = grad(grad_vector, params, grad_outputs=vector)
        hvp_batch = torch.cat([h.reshape(-1) for h in hvp_batch if h is not None])

        if hvp_result is None:
            hvp_result = (hvp_batch * X_batch.shape[0])
        else:
            hvp_result += (hvp_batch * X_batch.shape[0])
        total_item += X_batch.shape[0]
    return hvp_result / total_item


# Conjugate gradient method for solving H^-1 * grad
def conjugate_gradient(hvp_fn, b, tol=1e-6, max_iter=100):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = torch.dot(r, r)

    for i in range(max_iter):
        Ap = hvp_fn(p)
        alpha = rs_old / torch.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = torch.dot(r, r)
        print('norm of the conjugate gradient: {}'.format(torch.sqrt(rs_new)))

        if torch.sqrt(rs_new) < tol:
            print(f"Conjugate gradient converged in {i + 1} iterations.")
            break

        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x


def main():
    stime = time.time()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MNIST dataset
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    # test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # transform = get_transforms('cifar10')
    # transform = v2.Compose([
    #         v2.Resize((128, 128), interpolation=InterpolationMode.BILINEAR),  # ResNet18 expects 224x224 input size
    #         v2.ToImage(),
    #         v2.ToDtype(torch.float32, scale=True),
    #         v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization for pretrained models
    #     ])
    transform = AlexNet_Weights.IMAGENET1K_V1.transforms()
    train_dataset, test_dataset = get_train_test_datasets('cifar100', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Select a subset of 5% of the training data
    forget_size = int(len(train_dataset) * 0.05)
    # retain_size = len(train_dataset) - forget_size
    indices = torch.randperm(len(train_dataset))
    findices, rindices = indices[:forget_size], indices[forget_size:]
    # print(len(findices), len(rindices))
    forget_dataset = Subset(train_dataset, findices)
    retain_dataset = Subset(train_dataset, rindices)
    forget_loader = DataLoader(forget_dataset, batch_size=128, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=128, shuffle=True)

    # Initialize model, custom loss, and optimizer
    # model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_classes = 100
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = alexnet(weights=AlexNet_Weights.DEFAULT)
    model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
    model = model.to(device)
    # model = LeNet5().to(device)
    custom_loss = L2RegularizedCrossEntropyLoss(l2_lambda=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Training the model...")
    for epoch in range(50):
        model.train()
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/50"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = custom_loss(outputs, y_batch, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        evaluate_model(model, test_loader, custom_loss, device, log=False)
    print("Training complete!")

    # Evaluate the trained model
    print("Evaluating the trained model...")
    evaluate_model(model, train_loader, custom_loss, device, name="Whole Training Data")
    evaluate_model(model, test_loader, custom_loss, device, name="Test Data")
    evaluate_model(model, retain_loader, custom_loss, device, name="Retain Data")
    evaluate_model(model, forget_loader, custom_loss, device, name="Forget Data")

    # Compute the overall gradient on the subset
    print("Computing overall gradient...")
    params = list(model.parameters())
    overall_gradient = None
    total_size = 0
    for X_batch, y_batch in tqdm(forget_loader, desc="Computing Gradient on Forget"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = custom_loss(outputs, y_batch, model)
        gradients = grad(loss, params)
        grad_vector = torch.cat([g.view(-1) for g in gradients if g is not None])

        if overall_gradient is None:
            overall_gradient = (grad_vector * X_batch.shape[0])
        else:
            overall_gradient += (grad_vector * X_batch.shape[0])
        total_size += X_batch.shape[0]
    overall_gradient /= total_size

    overall_gradient = -(len(forget_dataset) / len(retain_dataset)) * overall_gradient

    # Define HVP function for the subset
    def hvp_fn_train(v):
        return batched_hvp(model, train_loader, custom_loss, params, v, device)

    def hvp_fn_forget(v):
        return batched_hvp(model, forget_loader, custom_loss, params, v, device)

    # Solve H^-1 * grad using conjugate gradient
    print("Solving for H^-1 * grad...")
    htrain_inv_grad = conjugate_gradient(hvp_fn_train, overall_gradient).to('cpu')
    hforget_inv_grad = conjugate_gradient(hvp_fn_forget, overall_gradient).to('cpu')
    h_inv_grad = (len(train_dataset) * htrain_inv_grad - len(forget_dataset) * hforget_inv_grad) / (
                len(train_dataset) - len(forget_dataset))
    model = model.to('cpu')

    # Clone and update the model
    # updated_model = resnet18()
    updated_model = alexnet(weights=AlexNet_Weights.DEFAULT)
    updated_model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
    updated_model.load_state_dict(model.state_dict())

    print("Updating model parameters using H^-1 * grad...")
    offset = 0
    for param in updated_model.parameters():
        param_size = param.numel()
        param_update = h_inv_grad[offset: offset + param_size].view(param.size())
        param.data -= param_update
        offset += param_size

    # Evaluate the updated model
    print("\nEvaluating the updated model...")
    updated_model = updated_model.to(device)
    evaluate_model(updated_model, train_loader, custom_loss, device, name="Whole Training Data")
    evaluate_model(updated_model, test_loader, custom_loss, device, name="Test Data")
    evaluate_model(updated_model, retain_loader, custom_loss, device, name="Retain Data")
    evaluate_model(updated_model, forget_loader, custom_loss, device, name="Forget Data")
    etime = time.time()
    logging.info('elapsed time: {:.3f} mins'.format((etime - stime) / 60))


if __name__ == "__main__":
    main()
