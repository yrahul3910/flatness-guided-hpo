import torch
import traceback
import numpy as np
import os
from torchvision.transforms import v2
from torchvision.datasets import MNIST, SVHN
from torch.utils.data import DataLoader
from backpack.hessianfree.hvp import hessian_vector_product
from backpack import extend


BATCH_SIZE = 64


class ImageModel(torch.nn.Module):
    def __init__(self, dataset: str = "mnist", learning_rate=1e-3):
        super().__init__()

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
    rand = ((torch.rand(shape) < 0.5)) * 2 - 1
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(dataset, num_workers=7, batch_size=BATCH_SIZE)
loss_fn = extend(torch.nn.CrossEntropyLoss().to(device))
model = extend(
    torch.nn.DistributedDataParallel(ImageModel(dataset=dataset_name)).to(device)
)
optim = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1

try:
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_fghpo = 0.0
        total_trace = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optim.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            loss.backward(retain_graph=True)

            cur_trace = hutchinson_trace_autodiff_blockwise(model, loss, 20, device)
            cur_fghpo = stcvx(model, train_loader, device)
            print("Estimated trace:", cur_trace)
            print("FGHPO estimate:", cur_fghpo)
            optim.step()

            total_loss += loss.item()
            total_fghpo += cur_fghpo
            total_trace += cur_trace

        model.eval()

        avg_loss = total_loss / len(train_loader)
        print("-" * 20)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        print(f"Total FGHPO estimate: {total_fghpo}")
        print(f"Total trace estimate: {total_trace}")
except Exception:
    traceback.print_exc()
    print("-" * 20)
    print(f"Total FGHPO estimate: {total_fghpo}")
    print(f"Total trace estimate: {total_trace}")
