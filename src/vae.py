import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


def reparameterize(mu, logvar):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


def loss_function(recons, x, mu, log_var,
                  **kwargs) -> dict:
    kld_weight = 0.025  # Account for the minibatch samples from the dataset
    recons_loss = F.mse_loss(recons, x)

    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

    loss = recons_loss + kld_weight * kld_loss
    return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}


class VanillaVAE(nn.Module):
    def __init__(self,
                 in_dim: int,
                 latent_dim: int,
                 hidden_dims) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        curr_dim = in_dim
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(curr_dim, h_dim),
                    nn.ReLU())
            )
            curr_dim = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.ReLU())
            )

        modules.append(nn.Linear(hidden_dims[-1], in_dim))
        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param x: (Tensor)
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor)
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        return result

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = reparameterize(mu, log_var)
        return [self.decode(z), x, mu, log_var]

    def sample(self,
               num_samples: int,
               current_device: int):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor)
        :return: (Tensor)
        """

        return self.forward(x)[0]


def log_prob_x(x, vae: VanillaVAE, k=50, sigma=1.0):
    """
    Estimate log p_theta(x) using importance sampling.
    """
    vae.eval()
    with torch.no_grad():
        # Get approximate posterior parameters
        mu, logvar = vae.encode(x)
        std = torch.exp(0.5 * logvar)

        # Sample from approximate posterior q(z|x)
        z_samples = mu + std * torch.randn(k, mu.size(1)).to(x.device)

        # Prior log probability p(z)
        log_pz = -0.5 * torch.sum(z_samples ** 2, dim=1) - 0.5 * z_samples.size(1) * torch.log(
            torch.tensor(2 * torch.pi))

        # Approximate posterior log probability q(z|x)
        log_qz_given_x = -0.5 * torch.sum(((z_samples - mu) / std) ** 2, dim=1) - 0.5 * torch.sum(
            torch.log(2 * torch.pi * logvar.exp()), dim=1)

        # Likelihood log p(x|z)
        recon_x = vae.decode(z_samples)
        log_px_given_z = -torch.sum((x - recon_x) ** 2, dim=1) / (2 * sigma ** 2) - 0.5 * x.size(1) * torch.log(
            torch.tensor(2 * torch.pi * sigma ** 2))

        # Importance weights
        log_w = log_px_given_z + log_pz - log_qz_given_x

        # Use log-sum-exp for stability
        max_log_w = torch.max(log_w)
        log_px = max_log_w + torch.log(torch.sum(torch.exp(log_w - max_log_w))) - torch.log(torch.tensor(k))
        return log_px.item()


def entropy(vae, dataloader, k=500, sigma=1.0, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Estimate Entropy
    vae.eval()
    log_probs = []
    pbar = tqdm(dataloader, desc='entropy estimation', unit='batch')
    for batch in pbar:
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch, _ = batch
        for x in batch:
            log_prob = log_prob_x(x.unsqueeze(0).to(device), vae, k=k, sigma=sigma)
            log_probs.append(log_prob)

    # Compute entropy
    entropy_estimate = -torch.mean(torch.tensor(log_probs))
    return entropy_estimate
