import os
import pickle

import lightning as L
import torch
import torch.nn.functional as F
from torch import autograd
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import v2


def compute_hessian_frobenius_norm(model, loss_fn, data_batch):
    """Compute the Frobenius norm of the Hessian of the loss with respect to
    the penultimate layer's weights.

    Args:
        model: PyTorch model
        loss_fn: Loss function
        data_batch: Tuple of (inputs, targets) for computing loss

    Returns:
        float: Frobenius norm of the Hessian matrix

    """
    # Ensure model is in eval mode
    model.eval()

    # Get penultimate layer
    parameterized_layers = [module for module in model.modules()
                           if list(module.parameters())]
    penultimate_layer = parameterized_layers[-2]
    weights = next(penultimate_layer.parameters())

    inputs, targets = data_batch

    # Ensure inputs and targets are properly formatted and on the correct device
    device = weights.device
    inputs = inputs.to(device).float()  # Convert to float and move to device
    if len(inputs.shape) == 3:  # Add channel dimension if needed
        inputs = inputs.unsqueeze(1)
    targets = targets.to(device).long()  # Convert to long and move to device

    # Take a smaller subset for computation efficiency
    batch_size = min(1000, inputs.shape[0])
    inputs = inputs[:batch_size]
    targets = targets[:batch_size]

    def loss_wrapper(weights):
        weights_param = weights.detach().clone().requires_grad_(True)

        # Temporarily replace the layer's weights
        original_weights = penultimate_layer.weight.data.clone()
        penultimate_layer.weight.data = weights_param

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.requires_grad = True

        # Restore original weights
        penultimate_layer.weight.data = original_weights
        return loss

    def hvp(v):
        """Compute Hessian-vector product."""
        def grad_output(weights):
            loss, weights_with_grad = loss_wrapper(weights)
            grad, = autograd.grad(loss, weights_with_grad, create_graph=True)
            return grad

        return autograd.grad(
            torch.sum(grad_output(weights) * v),
            weights
        )[0]

    # Estimate Frobenius norm using Hutchinson's trace estimator
    def hutchinson_estimator(num_samples=50):  # Reduced number of samples for efficiency
        total = 0
        for _ in range(num_samples):
            z = torch.randn_like(weights)
            hz = hvp(z)
            total += torch.sum(hz * hz)
        return torch.sqrt(total / num_samples)

    # Compute the Frobenius norm
    with torch.no_grad():
        frob_norm = hutchinson_estimator()

    return frob_norm.item()

class MNISTModel(L.LightningModule):
    def __init__(self, learning_rate=1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Define the network architecture
        self.conv_block_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32)
        )

        self.conv_block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64)
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

dataset = MNIST(os.getcwd(), download=True, transform=v2.Compose([v2.ToTensor(), v2.ToDtype(torch.float32)]))

train_loader = DataLoader(dataset, num_workers=7, batch_size=128)
model = MNISTModel()

trainer = L.Trainer(max_epochs=10)
trainer.fit(model=model, train_dataloaders=train_loader)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Calculate Frobenius norm with proper data preprocessing
frob_norm = compute_hessian_frobenius_norm(
    model,
    torch.nn.CrossEntropyLoss(),
    (dataset.data, dataset.targets)
)
