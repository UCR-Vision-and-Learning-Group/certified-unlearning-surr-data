import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.autograd import grad
from tqdm import tqdm

from src.loss import L2RegularizedCrossEntropyLoss
from src.data import get_train_test_datasets, get_transforms
from torchvision.models import resnet18, ResNet18_Weights
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='hinvgrad_resnet18_cifar10.log',
    filemode='w'
)


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

# Function to evaluate a model
def evaluate_model(model, dataloader, criterion, device, name="Dataset"):
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
def conjugate_gradient(hvp_fn, b, tol=1e-6, max_iter=1):
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
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MNIST dataset
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    # test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    transform = get_transforms('cifar10')
    train_dataset, test_dataset = get_train_test_datasets('cifar10', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Select a subset of 5% of the training data
    forget_size = int(len(train_dataset) * 0.05)
    # retain_size = len(train_dataset) - forget_size
    indices = torch.randperm(len(train_dataset))
    findices, rindices = indices[:forget_size], indices[forget_size:]
    # print(len(findices), len(rindices))
    forget_dataset = Subset(train_dataset, findices)
    retain_dataset = Subset(train_dataset, rindices)
    forget_loader = DataLoader(forget_dataset, batch_size=64, shuffle=True)
    retain_loader = DataLoader(retain_dataset, batch_size=64, shuffle=True)

    # Initialize model, custom loss, and optimizer
    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    # model = LeNet5().to(device)
    custom_loss = L2RegularizedCrossEntropyLoss(l2_lambda=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Training the model...")
    for epoch in range(1):
        model.train()
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/10"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = custom_loss(outputs, y_batch, model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
    updated_model = resnet18()
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


if __name__ == "__main__":
    main()
