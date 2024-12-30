import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.distributions.uniform import Uniform
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets

import numpy as np
from scipy import stats
import math
from functools import partial
from typing import Callable
import matplotlib
import matplotlib.pyplot as plt
import cmasher as cmr


class DataGeneratorBase:
    def __init__(self):
        pass

    def rvs(self, size=(1,)):
        raise NotImplementedError()

    def entropy(self):
        raise NotImplementedError()

    def get_Lipschitz(self, L_base):
        raise NotImplementedError('Lipschitz constant of base_distribution not known')

    def get_p_max(self, p_max_base):
        raise NotImplementedError('P_max of base_distribution not known')

    def plot(self):
        raise NotImplementedError('Plot of base_distribution not known')

    def pdf(self, x):
        xs, ys = self.plot()
        return np.interp(x=x, xp=xs, fp=ys)

    def logpdf(self, x):
        return np.log(self.pdf(x))


class DataGeneratorMulti(DataGeneratorBase):
    base_generator: DataGeneratorBase
    d: int

    def __init__(self, base_generator: DataGeneratorBase, d: int):
        self.d = d
        self.base_generator = base_generator
        super(DataGeneratorMulti, self).__init__()

    def rvs(self, size=(1,)):
        return self.base_generator.rvs(size + (self.d,))

    def entropy(self):
        return self.d * self.base_generator.entropy()

    def plot(self):
        return self.base_generator.plot()

    def pdf(self, x):
        y = self.base_generator.pdf(x)
        if isinstance(x, torch.TensorType):
            p = torch.prod(y, dim=-1)
        else:
            p = np.prod(y, axis=-1)
        return p

    def logpdf(self, x):
        y = self.base_generator.logpdf(x)
        if isinstance(x, torch.TensorType):
            p = torch.sum(y, dim=-1)
        else:
            p = np.sum(y, axis=-1)
        return p


class DataGeneratorPiecewise(DataGeneratorBase):
    def __init__(self,
                 base_distribution: Callable,
                 weights: np.ndarray,
                 starts: np.ndarray,
                 scales: np.ndarray,
                 interval=(0.0, 1.0)):
        super(DataGeneratorPiecewise, self).__init__()

        self.base_distribution = base_distribution
        self.interval = interval

        weights = np.array(weights)
        starts = np.array(starts)
        scales = np.array(scales)

        assert weights.shape == starts.shape == scales.shape
        n_mixture = weights.shape[0]

        self.starts = starts
        self.weights = weights
        self.scales = scales

        self.selection_dist = stats.rv_discrete(b=n_mixture,
                                                values=(range(n_mixture), self.weights))
        self._check_no_overlap()

    def rvs(self, size=(1,)):
        selection = self.selection_dist.rvs(size=size)
        offsets = self.starts[selection]
        scales = self.scales[selection]
        distr = self.base_distribution(scale=scales)
        base_rv = distr.rvs(size=size)
        return offsets + base_rv

    def entropy(self):
        base_entropy = (self.base_distribution(scale=self.scales).entropy() * self.weights).sum()
        selection_entropy = self.selection_dist.entropy()

        return base_entropy + selection_entropy

    def _check_no_overlap(self):
        # Check
        idx = self.starts.argsort()
        ends = self.starts[idx] + self.scales[idx]
        starts = self.starts[idx]
        assert np.all(ends[:-1] <= starts[1:]), "some mixtures overlap"


class TriangleGenerator(DataGeneratorPiecewise):
    def __init__(self, *args, **kwargs):
        base_distribution = partial(stats.triang, c=.5)
        super(TriangleGenerator, self).__init__(*args, base_distribution=base_distribution, **kwargs)

    def plot(self):
        xss = list()
        yss = list()
        for weight, scale, start in zip(self.weights, self.scales, self.starts):
            xs = [start, start + scale / 2, start + scale]
            ys = [0.0, 2 / scale * weight, 0.0]
            xss = xss + xs
            yss = yss + ys

        return np.array(xss), np.array(yss)


class AffineGenerator(DataGeneratorBase):
    a: np.ndarray
    a0: np.ndarray
    b: np.ndarray
    b0: np.ndarray
    b0i: np.ndarray
    da: np.ndarray
    db: np.ndarray
    dbi: np.ndarray
    p: np.ndarray
    uniform: np.ndarray
    _entropy: np.ndarray = None
    EPSILON = 1e-9

    def __init__(self, a: np.ndarray, b: np.ndarray):
        super(AffineGenerator, self).__init__()

        self.a = np.array(a).reshape((-1))
        self.b = np.array(b).reshape((-1))
        idx = np.argsort(self.a)
        self.a = self.a[idx]
        b = self.b[idx]

        assert self.a.shape == self.b.shape

        a0 = self.a[:-1]
        a1 = self.a[1:]
        b0 = b[:-1]
        b1 = b[1:]

        self.a0 = a0
        self.da = a1 - a0

        c = np.sum(self.da * (b1 + b0) / 2.0)
        self.b = b / c
        b0 = self.b[:-1]
        b1 = self.b[1:]
        self.b0 = b0
        self.b1 = b1
        self.db = b1 - b0

        # Take care of slope = 0.0
        self.uniform = np.abs(self.db) < self.EPSILON
        self.db[self.uniform] = 0.0
        self.b0[self.uniform] = (b0[self.uniform] + b1[self.uniform]) / 2.0
        self.b1[self.uniform] = self.b0[self.uniform]

        self.p = self.da * (b1 + b0) / 2.0

        self.b0i = self.b0 / self.p
        self.dbi = self.db / self.p

        n_mixture = self.p.shape[0]
        self.selection_dist = stats.rv_discrete(b=n_mixture,
                                                values=(range(n_mixture), self.p))

    def rvs(self, size=(1,)):
        selection = self.selection_dist.rvs(size=size)
        u = self.uniform[selection]

        b0 = self.b[:-1] / self.p
        a0 = self.a[:-1]
        b0 = b0[selection]
        a0 = a0[selection]
        da = self.da[selection]
        db = (self.db / self.p)[selection]

        y = np.random.uniform(size=size)
        #y = np.zeros(size)
        with np.errstate(divide='ignore', invalid='ignore'):
            x0 = - da * b0 / db + np.sqrt((da * b0 / db) ** 2 + (2.0 * da / db) * y)
            x1 = - da * b0 / db - np.sqrt((da * b0 / db) ** 2 + (2.0 * da / db) * y)
        x0[u] = da[u] * y[u]
        x1[u] = x0[u]

        assert not np.any(np.isnan(x0))
        assert not np.any(np.isnan(x1))

        x = -np.ones_like(x0)
        idx = np.logical_and(x0 < 0, x1 >= 0)
        x[idx] = x1[idx]
        idx = np.logical_and(x0 >= 0, x1 < 0)
        x[idx] = x0[idx]
        idx = np.logical_and(x0 >= 0, x1 >= 0)
        x[idx] = np.minimum(x0, x1)[idx]
        assert not np.any(np.logical_and(x0 < 0, x1 < 0))
        return a0 + x

    def entropy(self):
        if self._entropy is None:
            b0 = self.b0
            b1 = self.b1
            with np.errstate(divide='ignore', invalid='ignore'):
                di0 = b0 ** 2 * np.log(b0)
                di1 = b1 ** 2 * np.log(b1)
                di0[np.abs(b0) < self.EPSILON] = 0.0
                di1[np.abs(b1) < self.EPSILON] = 0.0
                self._entropy = np.nansum(self.p - self.da * (di1 - di0) / self.db) / 2.0
            self._entropy += -np.sum(self.da[self.uniform] * b0[self.uniform] * np.log(b0[self.uniform]))
        return self._entropy

    def plot(self):
        return self.a, self.b


def get_random_data_generator(base_pdf: str, number=3, seed=7):
    rng = np.random.default_rng(seed)

    if base_pdf == 'triangle':
        base_generator = TriangleGenerator
        weights = rng.uniform(size=(number - 1,))
        weights.sort()
        weights = np.diff(np.concatenate(([0.0], weights, [1.0]), axis=0))
        starts = np.ndarray((number,))
        scales = np.ndarray((number,))
        for i in range(number):
            scales[i] = 0.1 + 0.9 * rng.uniform()
            starts[i] = i

        return base_generator(weights=weights,
                              starts=starts,
                              scales=scales)
    elif base_pdf == 'affine':
        base_generator = AffineGenerator
        a = np.ndarray((number + 1,))
        b = np.ndarray((number + 1,))
        x = 0.0
        y = 0.0
        for i in range(number):
            a[i] = x
            b[i] = y
            y = rng.uniform()
            x += rng.uniform()
        a[number] = x
        b[number] = 0.0
        return base_generator(a=a, b=b)


##########################
# 2 RV for MI estimation #
##########################


class XZN:
    def __init__(self, dim, device):
        self.dim = dim
        self.device = device

    def I(self, rho):
        raise NotImplementedError()

    def ItoRho(self, I: float) -> float:
        raise NotImplementedError()

    def dI(self, rho):
        raise NotImplementedError()

    def hY(self, rho):
        raise NotImplementedError()

    def logi(self, rho, x, y):
        raise NotImplementedError()

    def draw_samples(self, num_samples):
        raise NotImplementedError()


class UniformsXZN(XZN):
    def __init__(self, *args, **kwargs):
        super(UniformsXZN, self).__init__(*args, **kwargs)
        self.pdf = torch.rand

    def I(self, rho):
        HN = torch.log(2 * torch.sqrt(3 - 3 * rho ** 2))
        HY = self.hY(rho)
        return HY - self.dim * HN

    def ItoRho(self, I: float) -> float:
        rho = 2 * I * np.sqrt((4 * I ** 2 + (-4 * I - np.log(16) - np.log(9)) * np.log(
            12) + 8 * I * np.log(2) + np.log(3) * (4 * I + np.log(16)) + 1 + np.log(
            3) ** 2 + 4 * np.log(2) ** 2 + np.log(12) ** 2) ** (-1))
        return rho

    def draw_samples(self, num_samples):
        X = self.pdf((num_samples, self.dim), device=self.device)
        ep = self.pdf((num_samples, self.dim), device=self.device)
        X = 2 * np.sqrt(3) * (X - 1 / 2)
        ep = 2 * np.sqrt(3) * (ep - 1 / 2)
        return X, X, ep

    def logi(self, rho, x, y):
        delta_N = torch.sqrt((1 - rho ** 2) * 3)
        delta_X = torch.sqrt(rho ** 2 * 3)
        d_1 = delta_N + delta_X
        d_2 = torch.abs(delta_N - delta_X)
        A = 1 / (d_1 + d_2)
        delta = d_1 - d_2
        k = A / delta

        y_abs = y.abs()
        log_py = torch.log(A - k * torch.relu(y_abs - d_2))

        lgi = - np.log(2) - torch.log(delta_N) - log_py
        lgi = lgi.sum(dim=-1)
        return lgi

    def hY(self, rho):
        rho = torch.maximum(rho, torch.sqrt(1 - rho ** 2))
        HY = (1 / (2 * rho)) * (math.sqrt(1 - rho ** 2) + np.log(3) * (-rho) + np.log(6) * 2 * rho +
                                torch.log(rho) * 2 * rho)
        return self.dim * HY

    def dI(self, rho):
        if rho > torch.sqrt(1 - rho ** 2):
            dI = (-rho + (1 / 2) * torch.sqrt(1 - rho ** 2)) / (rho ** 2 * (rho ** 2 - 1))
        else:
            dI = (1 / 2) * torch.sqrt(1 - rho ** 2) / (rho ** 4 - 2 * rho ** 2 + 1)
        return self.dim * dI


class MultiTriangleXZN(XZN):
    def __init__(self, *args, **kwargs):
        super(MultiTriangleXZN, self).__init__(*args, **kwargs)
        self.pdf = Uniform(0.0, .5)
        self.components = 2
        self.categorical = Categorical(torch.ones((self.components,)) / self.components)

    def hY(self, rho):
        assert rho >= 1 / np.sqrt(2)
        h = np.log(self.components) + (1 / 2 + torch.log((1 - rho ** 2) / 4) / 2)
        return self.dim * h

    def dI(self, rho):
        assert rho <= 1.0
        return torch.tensor([0.0]).to(device=self.device)

    def I(self, rho):
        assert rho <= 1.0
        IXY = torch.tensor([np.log(self.components)], device=self.device)
        return self.dim * IXY

    def draw_samples(self, num_samples):
        Z = self.categorical.sample((num_samples, self.dim)).to(device=self.device)
        ep0 = self.pdf.sample((num_samples, self.dim)).to(device=self.device) + self.pdf.sample(
            (num_samples, self.dim)).to(device=self.device)
        ep = self.pdf.sample((num_samples, self.dim)).to(device=self.device) + self.pdf.sample(
            (num_samples, self.dim)).to(device=self.device)
        return Z + ep0, Z, ep


class GaussianXZN(XZN):
    def __init__(self, *args, **kwargs):
        super(GaussianXZN, self).__init__(*args, **kwargs)
        self.pdf = MultivariateNormal(torch.zeros(self.dim).to(self.device),
                                      torch.eye(self.dim).to(self.device))

    def ItoRho(self, I):
        rho = np.sqrt(1 - np.exp(-2 * I))
        print(f"I {I} --> rho {rho}")
        return rho

    def hY(self, rho):
        return self.dim / 2 * (np.log(2 * np.pi) + 1)

    def I(self, rho):
        num_nats = - self.dim / 2 * torch.log(1 - rho ** 2)
        return num_nats

    def dI(self, rho):
        num_nats = rho * self.dim / (1 - rho ** 2)
        return num_nats

    def logi(self, rho, x, y):
        C = -self.dim / 2 * torch.log(1 - rho ** 2)
        li = -1 / (2 * (1 - rho ** 2)) * (rho ** 2 * (x ** 2 + y ** 2) - 2 * rho * x * y)
        return C + torch.sum(li, dim=-1)

    def draw_samples(self, num_samples):
        X, ep = torch.split(self.pdf.sample((2 * num_samples,)), num_samples)
        return X, X, ep


class XY(object):
    def __init__(self, xyn, rho, cubed=False):
        self.xyn: XZN = xyn
        self.rho = rho
        self.cubed = cubed

    def I(self):
        return self.xyn.I(self.rho)

    def hY(self):
        return self.xyn.hY(self.rho)

    def dI(self):
        return self.xyn.dI(self.rho)

    def _get_samples(self):
        return self.x, self.z * self.rho + torch.sqrt(1 - self.rho ** 2) * self.n

    def repeat_samples(self):
        x, y = self._get_samples()
        if self.cubed:
            x = x ** 3
        return x, y
        #return y, x

    def draw_samples(self, num_samples):
        self.x, self.z, self.n = self.xyn.draw_samples(num_samples=num_samples)
        return self.repeat_samples()

    def logi(self):
        x, y = self._get_samples()
        return self.xyn.logi(self.rho, x, y)


#################### CODE STARTS HERE #####################################
cmap_low = matplotlib.cm.copper_r
cmap_high = cmr.get_sub_cmap('Greens', 0, 0.5)


def plot_samples(samples, n_rows=4, n_cols=5, low_end=0, high_end=1, extra_range_frac=0.25, cmap_low=cmap_low,
                 cmap_high=cmap_high):
    side = int(np.sqrt(samples.shape[1]))
    extra_range = extra_range_frac * (high_end - low_end)

    assert samples.shape[0] == n_rows * n_cols, f"Expected {n_rows * n_cols} samples, got {samples.shape[0]}"

    samples = samples.reshape(-1, side, side)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    for i, ax in enumerate(axes.flatten()):
        if i >= samples.shape[0]:
            break

        image = samples[i]
        mask_low = samples[i] > low_end
        mask_high = samples[i] < high_end
        masked_image_low = np.ma.masked_array(samples[i], mask=mask_low)
        masked_image_high = np.ma.masked_array(samples[i], mask=mask_high)

        im = ax.imshow(image, cmap="gray", vmin=low_end, vmax=high_end, interpolation='none')
        ax.imshow(masked_image_low, cmap=cmap_low, vmin=low_end - extra_range, vmax=low_end)
        ax.imshow(masked_image_high, cmap=cmap_high, vmin=high_end, vmax=high_end + extra_range)

        ax.axis('off')


class TriangleDataset(Dataset):
    def __init__(self, n_samples, device='cpu', number=3, dim=1, seed=7):
        self.n_samples = n_samples
        self.number = number
        self.dim = dim
        self.seed = seed

        self.gen1 = get_random_data_generator('triangle', number=number, seed=seed)
        self.gen = DataGeneratorMulti(self.gen1, dim)
        self.samples = torch.tensor(self.gen.rvs((n_samples,))).float().to(device)

    def entropy(self):
        return self.gen.entropy()

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample


class BallDataset(Dataset):
    def __init__(self, n_samples, device='cpu', dim=1, r=1.0, seed=7):
        self.n_samples = n_samples
        self.dim = dim
        self.r = r

        generator = torch.Generator().manual_seed(seed)
        self.samples = torch.randn(n_samples, dim, generator=generator)
        self.samples /= torch.norm(self.samples, dim=1).reshape(-1, 1)
        self.samples *= torch.rand(n_samples, 1, generator=generator) ** (1 / dim)
        self.samples *= r
        self.samples = self.samples.float().to(device)

    def volume(self):
        return np.pi ** (self.dim / 2) / np.math.gamma(self.dim / 2 + 1) * self.r ** self.dim

    def entropy(self):
        return np.log(self.volume())

    def cross_entropy(self, sigma2):
        '''
        Computes the cross entropy C(B, N(0, sigma2 * I_d))
        '''
        return math.log((2 * math.pi * sigma2) ** (self.dim / 2)) + 1 / (2 * sigma2) * self.dim / (
                self.dim + 2) * self.r ** 2

    def relative_entropy(self, sigma2):
        '''
        Computes the relative entropy R(B || N(0, sigma2 * I_d))
        '''
        # return math.log(math.gamma(self.dim/2 + 1) / (math.pi**(self.dim/2) * self.r**self.dim)) + math.log((
        # 2*math.pi*sigma2)**(self.dim/2)) + 1/(2*sigma2) * self.dim / (self.dim+2) * self.r**2#(1 / self.r**2)

        return -self.entropy() + self.cross_entropy(sigma2)

    def variance(self):
        '''
        Theoretical (marginal) variance of a sample from the ball
        '''
        return self.r ** 2 / (self.dim + 2)

    def emp_variance(self):
        '''
        Empirical (marginal) variance of the samples
        '''
        return self.samples.cpu().numpy().var(axis=0).mean()

    @staticmethod
    def radius_ball(d):
        '''
        Radius such that volume of ball is 1
        '''
        return (math.gamma(d / 2 + 1) / math.pi ** (d / 2)) ** (1 / d)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample


class HypercubeDataset(Dataset):
    """
    A dataset of samples from the d-dimensional hypercube [0, side]^d
    """

    def __init__(self, n_samples, device='cpu', dim=1, side=1.0, seed=7):
        self.n_samples = n_samples
        self.dim = dim
        self.side = side

        self.samples = torch.rand(n_samples, dim, generator=torch.Generator().manual_seed(seed))
        self.samples *= side
        self.samples = self.samples.float().to(device)

    def volume(self):
        return self.side ** self.dim

    def entropy(self):
        return np.log(self.volume())

    def cross_entropy(self, sigma2):
        '''
        Computes the cross entropy C(C, N(1/2, sigma2 * I_d))
        '''
        #return math.log((2*math.pi*sigma2)**(self.dim/2)) + 1/(2*sigma2) * self.dim / (self.dim+2) * self.side**2

        return self.dim / 2 * (math.log(2 * math.pi * sigma2) + 1 / 12 * self.side ** 2 / sigma2)

    def relative_entropy(self, sigma2):
        '''
        Computes the relative entropy R(C || N(1/2, sigma2 * I_d))
        '''
        return -self.entropy() + self.cross_entropy(sigma2)

    def variance(self):
        '''
        Theoretical (marginal) variance of a sample from the hypercube
        '''
        return self.side ** 2 / 12

    def emp_variance(self):
        '''
        Empirical (marginal) variance of the samples
        '''
        return self.samples.cpu().numpy().var(axis=0).mean()

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample


class MoonsDataset(Dataset):
    def __init__(self, n_samples, noise, device='cpu', seed=7):
        self.samples, _ = datasets.make_moons(n_samples=n_samples, noise=noise, random_state=seed)
        self.samples = torch.tensor(self.samples).float().to(device)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample


class GaussianDataset(Dataset):
    def __init__(self, n_samples, n_class, mean, cov, seed=42):
        self.n_samples = n_samples
        self.mean = mean
        self.cov = cov
        self.n_class = n_class
        self.seed = seed

        self.X, self.y, metric_results = self._generate_multiclass_gaussian_dataset()
        self.disentanglement_clf, self.disentanglement_score = metric_results

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.n_samples

    def _generate_multiclass_gaussian_dataset(self):
        """
        Generate a high-dimensional multi-class Gaussian dataset with linearly separable labels.
        """

        X = np.random.multivariate_normal(self.mean, self.cov, self.n_samples)

        # Step 2: Define boundaries for the given number of classes
        def assign_label(x):
            angle = np.arctan2(x[1], x[0])  # Compute the angle of the first two dimensions
            class_angle = 2 * np.pi / self.n_class  # Angle span for each class
            return int((angle + np.pi) // class_angle) % self.n_class

        y = np.array([assign_label(x) for x in X])

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train a Logistic Regression model
        clf = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        disentanglement_score = accuracy
        metric_results = (clf, disentanglement_score)

        X = torch.tensor(X).float()
        y = torch.tensor(y).long()

        return X, y, metric_results

    def create_surr(self, mean, cov):
        surr_dataset = GaussianDataset(self.n_samples, self.n_class, mean, cov)
        y_hat = self.disentanglement_clf.predict(surr_dataset.X.numpy())
        surr_dataset.y = torch.tensor(y_hat).long()
        surr_dataset.disentanglement_clf = None
        surr_dataset.disentanglement_score = None
        return surr_dataset

    def calculate_kl_between(self, gauss_dataset):
        mean1, cov1 = self.mean, self.cov
        mean2, cov2 = gauss_dataset.mean, gauss_dataset.cov
        k = mean1.shape[0]  # Dimensionality
        cov2_inv = np.linalg.inv(cov2)
        trace_term = np.trace(cov2_inv @ cov1)
        mean_diff = mean2 - mean1
        quadratic_term = mean_diff.T @ cov2_inv @ mean_diff
        log_det_term = np.log(np.linalg.det(cov2)) - np.log(np.linalg.det(cov1))

        kl_div = 0.5 * (trace_term + quadratic_term - k + log_det_term)
        return kl_div


# class PoissonDataset(Dataset):
#     def __init__(self, n_samples, n_class, lambdas, seed=42):
#         self.n_samples = n_samples
#         self.n_class = n_class
#         self.lambdas = lambdas  # Array of lambda values for each class
#         self.seed = seed
#
#         self.X, self.y, metric_results = self._generate_multiclass_poisson_dataset()
#         self.disentanglement_clf, self.disentanglement_score = metric_results
#
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
#
#     def __len__(self):
#         return self.n_samples
#
#     def _generate_multiclass_poisson_dataset(self):
#         """
#         Generate a multi-class Poisson dataset.
#         """
#         np.random.seed(self.seed)
#         n_features = len(self.lambdas[0])
#
#         # Generate samples for each class
#         X = []
#         y = []
#         for class_id, lambda_vals in enumerate(self.lambdas):
#             class_samples = np.random.poisson(lambda_vals, size=(self.n_samples // self.n_class, n_features))
#             X.append(class_samples)
#             y.append(np.full((self.n_samples // self.n_class,), class_id))
#
#         X = np.vstack(X)
#         y = np.hstack(y)
#
#         # Split data into train and test sets
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
#         # Train a Logistic Regression model
#         clf = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
#         clf.fit(X_train, y_train)
#
#         # Evaluate on test set
#         y_pred = clf.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         disentanglement_score = accuracy
#         metric_results = (clf, disentanglement_score)
#
#         X = torch.tensor(X).float()
#         y = torch.tensor(y).long()
#
#         return X, y, metric_results
#
#     def create_surr(self, lambdas):
#         surr_dataset = PoissonDataset(self.n_samples, self.n_class, lambdas)
#         y_hat = self.disentanglement_clf.predict(surr_dataset.X.numpy())
#         surr_dataset.y = torch.tensor(y_hat).long()
#         surr_dataset.disentanglement_clf = None
#         surr_dataset.disentanglement_score = None
#         return surr_dataset
#
#     def calculate_kl_between(self, poisson_dataset):
#         kl_div = 0.0
#         for lambda1, lambda2 in zip(self.lambdas, poisson_dataset.lambdas):
#             kl_div += np.sum(np.array(lambda1) * (np.log(np.array(lambda1)) - np.log(np.array(lambda2))) - (np.array(lambda1) - np.array(lambda2)))
#         return kl_div