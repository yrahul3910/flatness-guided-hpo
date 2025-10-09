import os

import lightning as L
import numpy as np
import torch
from backpack import extend
from backpack.hessianfree.hvp import hessian_vector_product
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, SVHN
from torchvision.transforms import v2

BATCH_SIZE = 64


class ImageModel(L.LightningModule):
    def __init__(self, dataset: str = "mnist") -> None:
        super().__init__()
        self.loss_fn = extend(torch.nn.CrossEntropyLoss())

        if dataset == "mnist":
            n_channels = 1
            dims = 28
        else:
            n_channels = 3
            dims = 32

        # Define the network architecture
        self.conv_block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=n_channels,
                out_channels=32 * n_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32 * n_channels),
        )

        self.conv_block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32 * n_channels,
                out_channels=64 * n_channels,
                groups=n_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64 * n_channels),
        )

        self.fc1 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * n_channels * dims * dims, BATCH_SIZE),
            torch.nn.ReLU(),
        )

        self.fc2 = torch.nn.Sequential(torch.nn.Linear(BATCH_SIZE, 10))

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)

        x = self.fc1(x)
        self.al1 = x.detach().cpu().numpy()

        x = self.fc2(x)
        self.al = x.detach().cpu().numpy()

        return x

    def backward(self, loss) -> None:
        loss.backward(retain_graph=True)
        hutchinson_trace_autodiff_blockwise(model, loss, 20, self.device)
        stcvx(model, train_loader, self.device)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        return self.loss_fn(output, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def stcvx(model: ImageModel, dl: DataLoader, device):
    def Ka_func(xb):
        model(xb.to(device))
        return model.al, model.al1

    best_mu = -np.inf
    for xb, _ in dl:
        al, al1 = Ka_func(xb)
        Wl = model.state_dict()["fc2.0.weight"].cpu()
        mu = (
            np.linalg.norm(al) * np.linalg.norm(al1) / np.linalg.norm(Wl)
        ) / BATCH_SIZE
        if mu > best_mu and mu != np.inf:
            best_mu = mu

    return best_mu


def rademacher(shape, dtype=torch.float32, device="cpu"):
    """Sample from Rademacher distribution."""
    rand = (torch.rand(shape) < 0.5) * 2 - 1
    return rand.to(dtype).to(device)


def hutchinson_trace_autodiff_blockwise(model, loss, V, device):
    """Hessian trace estimate using autodiff block HVPs."""
    trace = 0

    for _ in range(V):
        for p in model.parameters():
            v = [rademacher(p.shape, device=device)]
            Hv = hessian_vector_product(loss, [p], v)
            vHv = torch.einsum("i,i->", v[0].flatten(), Hv[0].flatten())

            trace += vHv / V

    return trace


if __name__ == "__main__":
    dataset_name = "svhn"

    if dataset_name == "mnist":
        dataset = MNIST(
            os.getcwd(),
            download=True,
            transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)]),
        )
    else:
        dataset = SVHN(
            os.getcwd(),
            download=True,
            transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32)]),
        )

    trainer = L.Trainer(
        max_epochs=1, devices=torch.cuda.device_count(), accelerator="gpu"
    )

    train_loader = DataLoader(dataset, num_workers=4, batch_size=BATCH_SIZE)
    model = extend(ImageModel(dataset=dataset_name))

    trainer.fit(model=model, train_dataloaders=train_loader)
