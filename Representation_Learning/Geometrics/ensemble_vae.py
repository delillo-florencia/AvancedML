# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()

        elbo = torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """

    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model
                optimizer.zero_grad()
                # from IPython import embed; embed()
                loss = model(x)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break

import torch
import numpy as np

import torch

def curve_energy(c, decoder, latent_dim, num_points=100):
    t = torch.linspace(0, 1, num_points)
    dt = t[1] - t[0]
    
    curve_points = torch.stack([c(ti) for ti in t])
    curve_points.requires_grad_(True)
    
    # Modify the decoder output to return a tensor
    def decoder_tensor(x):
        dist = decoder(x)
        return dist.mean  # or another tensor representation
    
    jacobians = torch.autograd.functional.jacobian(decoder_tensor, curve_points)
    jacobians = jacobians.view(num_points, -1, latent_dim)
    
    velocity = torch.diff(curve_points, dim=0) / dt
    energy = torch.tensor(0.0, requires_grad=True)  # Initialize energy as a tensor with gradients
    for i in range(len(velocity)):
        J = jacobians[i]
        v = velocity[i]
        pullback_metric = torch.matmul(J.transpose(0, 1), J)
        energy = energy + torch.matmul(v.unsqueeze(0), torch.matmul(pullback_metric, v.unsqueeze(1))) * dt
    return energy
from torch.optim import Adam

def compute_geodesic(start, end, decoder, latent_dim, num_points=100, lr=0.01, num_iter=1000):
    """
    Compute the geodesic between two points in latent space.
    
    Parameters:
    - start: The starting point in latent space.
    - end: The ending point in latent space.
    - decoder: The decoder network of the VAE.
    - latent_dim: The dimensionality of the latent space.
    - num_points: The number of points to discretize the curve.
    - lr: Learning rate for the optimizer.
    - num_iter: Number of optimization iterations.
    
    Returns:
    - curve_points: The points along the geodesic.
    """
    # Initialize the curve as a straight line between start and end
    t = torch.linspace(0, 1, num_points)
    curve_points = torch.stack([start * (1 - ti) + end * ti for ti in t])
    curve_points.requires_grad_(True)
    print("Curved ok")
    
    # Optimizer
    optimizer = Adam([curve_points], lr=lr)
    
    for _ in range(num_iter):
        optimizer.zero_grad()
        
        # Define the curve function
        def c(ti):
            idx = int(ti * (num_points - 1))
            return curve_points[idx]
        
        # Compute the energy
        print("Computing energy...")
        energy = curve_energy(c, decoder, latent_dim, num_points)
        print("Enery ok, now backprop")
        # Backpropagate
        energy.backward()
        optimizer.step()
    
    return curve_points.detach()

import matplotlib.pyplot as plt

def plot_latent_space_with_geodesics(latent_variables, decoder, latent_dim, num_geodesics=25):
    """
    Plot the latent variables and geodesics between random pairs.
    
    Parameters:
    - latent_variables: The latent variables (e.g., from the VAE encoder).
    - decoder: The decoder network of the VAE.
    - latent_dim: The dimensionality of the latent space.
    - num_geodesics: The number of geodesics to plot.
    """
    plt.figure(figsize=(10, 10))
    
    # Plot the latent variables
    plt.scatter(latent_variables[:, 0], latent_variables[:, 1], alpha=0.5)
    
    # Compute and plot geodesics
    for _ in range(num_geodesics):
        start_idx, end_idx = np.random.choice(len(latent_variables), 2, replace=False)
        start = latent_variables[start_idx]
        end = latent_variables[end_idx]
        print("Computing geodesic...")
        
        geodesic = compute_geodesic(start, end, decoder, latent_dim)
        print("Geodesic found :)")
        plt.plot(geodesic[:, 0], geodesic[:, 1], 'r-', alpha=0.5)
    
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space with Geodesics')
    plt.savefig("latent_space_geodesics.png")
def sample_latent_space(dataloader, model, n_samples=25):
    """
    Sample points from the VAE's latent space.

    Parameters:
    dataloader: [torch.utils.data.DataLoader]
                Data loader for the dataset.
    model: [VAE]
           The trained VAE model.
    n_samples: [int]
               Number of samples to generate.

    Returns:
    latent_samples: [torch.Tensor]
                    A tensor of shape (n_samples, latent_dim) representing points in the latent space.
    """
    latent_samples = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            q = model.encoder(x)  # Encoder output
            z = q.rsample()  # Sample from the latent space
            latent_samples.append(z)
            if len(latent_samples) * x.size(0) >= n_samples:
                break

    # Concatenate all the sampled points
    latent_samples = torch.cat(latent_samples)[:n_samples]
    return latent_samples

def model_average_curve_energy(c, decoders, latent_dim, num_points=100, num_samples=10):
    """
    Compute the model-average curve energy using Monte Carlo approximation.
    
    Parameters:
    - c: A function that takes a parameter t in [0, 1] and returns a point in latent space.
    - decoders: A list of decoder networks (ensemble members).
    - latent_dim: The dimensionality of the latent space.
    - num_points: The number of points to discretize the curve for numerical integration.
    - num_samples: The number of Monte Carlo samples to approximate the expectation.
    
    Returns:
    - energy: The model-average energy of the curve.
    """
    t = torch.linspace(0, 1, num_points)
    dt = t[1] - t[0]
    
    # Compute the curve points
    curve_points = torch.stack([c(ti) for ti in t])
    
    # Compute the energy
    energy = 0.0
    for _ in range(num_samples):
        # Randomly sample two decoders from the ensemble
        f_i, f_k = np.random.choice(decoders, 2, replace=False)
        
        # Compute the Jacobians and velocities
        for i in range(len(t) - 1):
            point_i = curve_points[i].detach().requires_grad_(True)
            point_k = curve_points[i + 1].detach().requires_grad_(True)
            
            output_i = f_i(point_i)
            output_k = f_k(point_k)
            
            jacobian_i = torch.autograd.functional.jacobian(f_i, point_i)
            jacobian_k = torch.autograd.functional.jacobian(f_k, point_k)
            
            velocity = (point_k - point_i) / dt
            pullback_metric_i = torch.matmul(jacobian_i.transpose(0, 1), jacobian_i)
            pullback_metric_k = torch.matmul(jacobian_k.transpose(0, 1), jacobian_k)
            
            energy += torch.matmul(velocity.unsqueeze(0), torch.matmul(pullback_metric_i + pullback_metric_k, velocity.unsqueeze(1))).item() * dt
    
    return energy / num_samples
def compute_ensemble_geodesic(start, end, decoders, latent_dim, num_points=100, lr=0.01, num_iter=1000, num_samples=10):
    """
    Compute the geodesic between two points in latent space using an ensemble of decoders.
    
    Parameters:
    - start: The starting point in latent space.
    - end: The ending point in latent space.
    - decoders: A list of decoder networks (ensemble members).
    - latent_dim: The dimensionality of the latent space.
    - num_points: The number of points to discretize the curve.
    - lr: Learning rate for the optimizer.
    - num_iter: Number of optimization iterations.
    - num_samples: The number of Monte Carlo samples to approximate the expectation.
    
    Returns:
    - curve_points: The points along the geodesic.
    """
    # Initialize the curve as a straight line between start and end
    t = torch.linspace(0, 1, num_points)
    curve_points = torch.stack([start * (1 - ti) + end * ti for ti in t])
    curve_points.requires_grad_(True)
    
    # Optimizer
    optimizer = Adam([curve_points], lr=lr)
    
    for _ in range(num_iter):
        optimizer.zero_grad()
        
        # Define the curve function
        def c(ti):
            idx = int(ti * (num_points - 1))
            return curve_points[idx]
        
        # Compute the model-average energy
        energy = model_average_curve_energy(c, decoders, latent_dim, num_points, num_samples)
        
        # Backpropagate
        energy.backward()
        optimizer.step()
    
    return curve_points.detach()

def coefficient_of_variation(distances):
    """
    Compute the coefficient of variation (CoV) for a set of distances.
    
    Parameters:
    - distances: A list of distances (either Euclidean or geodesic).
    
    Returns:
    - cov: The coefficient of variation.
    """
    mean = torch.mean(torch.tensor(distances))
    std = torch.std(torch.tensor(distances))
    return (std / mean).item()

def evaluate_cov(vaes, test_pairs, latent_dim, num_geodesics=10):
    """
    Evaluate the coefficient of variation (CoV) for Euclidean and geodesic distances.
    
    Parameters:
    - vaes: A list of trained VAEs.
    - test_pairs: A list of fixed test point pairs.
    - latent_dim: The dimensionality of the latent space.
    - num_geodesics: The number of geodesics to compute for each pair.
    
    Returns:
    - cov_euclidean: The average CoV for Euclidean distances.
    - cov_geodesic: The average CoV for geodesic distances.
    """
    cov_euclidean = []
    cov_geodesic = []
    
    for (y_i, y_j) in test_pairs:
        euclidean_distances = []
        geodesic_distances = []
        
        for vae in vaes:
            x_i = vae.encode(y_i).mean
            x_j = vae.encode(y_j).mean
            
            # Euclidean distance
            euclidean_dist = torch.norm(x_i - x_j).item()
            euclidean_distances.append(euclidean_dist)
            
            # Geodesic distance
            geodesic = compute_ensemble_geodesic(x_i, x_j, vae.decoders, latent_dim)
            geodesic_dist = torch.sum(torch.norm(torch.diff(geodesic, dim=0), dim=0).item())
            geodesic_distances.append(geodesic_dist)
        
        cov_euclidean.append(coefficient_of_variation(euclidean_distances))
        cov_geodesic.append(coefficient_of_variation(geodesic_distances))
    
    return np.mean(cov_euclidean), np.mean(cov_geodesic)

def plot_latent_space_with_ensemble_geodesics(latent_variables, decoders, latent_dim, num_geodesics=25):
    """
    Plot the latent variables and geodesics between random pairs using an ensemble of decoders.
    
    Parameters:
    - latent_variables: The latent variables (e.g., from the VAE encoder).
    - decoders: A list of decoder networks (ensemble members).
    - latent_dim: The dimensionality of the latent space.
    - num_geodesics: The number of geodesics to plot.
    """
    plt.figure(figsize=(10, 10))
    
    # Plot the latent variables
    plt.scatter(latent_variables[:, 0], latent_variables[:, 1], alpha=0.5)
    
    # Compute and plot geodesics
    for _ in range(num_geodesics):
        start_idx, end_idx = np.random.choice(len(latent_variables), 2, replace=False)
        start = latent_variables[start_idx]
        end = latent_variables[end_idx]
        
        geodesic = compute_ensemble_geodesic(start, end, decoders, latent_dim)
        plt.plot(geodesic[:, 0], geodesic[:, 1], 'r-', alpha=0.5)
    
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space with Ensemble Geodesics')
    plt.show()
    
if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=3,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    # Choose mode to run
    if args.mode == "train":

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder,
            args.device,
        )
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model.pt",
        )

    elif args.mode == "sample":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0), "reconstruction_means.png"
            )

    elif args.mode == "eval":
        # Load trained model
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesics":

        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()
        latent_variables=sample_latent_space(mnist_test_loader, model, n_samples=4)
        print(latent_variables)
        plot_latent_space_with_geodesics(latent_variables,  model.decoder, latent_dim=2)

    elif args.mode == "geodesics_ensemble":
    # Load the ensemble of VAEs
        ensemble_size = 3  # Number of decoders in the ensemble
        vaes = []
        for i in range(ensemble_size):
            model = VAE(
                GaussianPrior(M),
                GaussianDecoder(new_decoder()),
                GaussianEncoder(new_encoder()),
            ).to(device)
            model.load_state_dict(torch.load(f"{args.experiment_folder}/model_{i}.pt"))
            model.eval()
            vaes.append(model)

        # Sample latent variables (use the same latent variables across all models)
        latent_variables=sample_latent_space(mnist_test_loader, vaes[0], n_samples=4)

        # Extract decoders from the ensemble
        decoders = [vae.decoder for vae in vaes]

        # Plot latent space with geodesics using the ensemble of decoders
        plot_latent_space_with_ensemble_geodesics(latent_variables, decoders, latent_dim=2)

        # Evaluate the coefficient of variation (CoV) for Euclidean and geodesic distances
        test_pairs = [(y_i, y_j) for y_i, y_j in zip(latent_variables[:10], latent_variables[10:20])]  # Example test pairs
        cov_euclidean, cov_geodesic = evaluate_cov(vaes, test_pairs, latent_dim=2)
        print(f"Average CoV for Euclidean distances: {cov_euclidean}")
        print(f"Average CoV for geodesic distances: {cov_geodesic}")
