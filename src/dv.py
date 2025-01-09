import torch
import torch.nn as nn
import torch.optim as optim
from itertools import cycle


# Define the function approximator for the variational function (f_theta)
class VariationalFunction(nn.Module):
    def __init__(self, input_dim):
        super(VariationalFunction, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# Donsker-Varadhan KL computation
def dv_kl_divergence(p_samples, q_samples, f_theta):
    f_p = f_theta(p_samples)  # Variational output for p_samples
    f_q = f_theta(q_samples)  # Variational output for q_samples

    # DV Bound
    dv_bound = f_p.mean() - torch.log(torch.exp(f_q).mean())
    return dv_bound


def train_dv_bound(ploader, qloader, device, num_epoch=10):
    (tmp,) = next(iter(qloader))
    shape = tmp.shape[1:]
    input_dim = 1
    for s in shape:
        input_dim *= s
    f_theta = VariationalFunction(input_dim).to(device)  # Move model to GPU
    optimizer = optim.Adam(f_theta.parameters(), lr=0.0001)
    qloader_cycle = cycle(qloader)
    for epoch in range(num_epoch):  # Adjust epochs as needed
        dv_loss_epoch = 0

        for (p_batch), (q_batch) in zip(ploader, qloader_cycle):
            if len(p_batch) > 1:
                (p_batch, _) = p_batch
            else:
                (p_batch,) = p_batch

            if len(q_batch) > 1:
                (q_batch, _) = q_batch
            else:
                (q_batch,) = q_batch

            p_batch, q_batch = p_batch.to(device), q_batch.to(device)  # Move data to GPU

            # Compute DV KL divergence
            dv_loss = -dv_kl_divergence(p_batch, q_batch, f_theta)  # Maximize DV bound

            # Backpropagation
            optimizer.zero_grad()
            dv_loss.backward()
            optimizer.step()

            dv_loss_epoch += dv_loss.item()

        print(f"Epoch {epoch + 1}, DV KL: {-dv_loss_epoch / len(ploader):.4f}")
    return f_theta, (-dv_loss_epoch / len(ploader))
