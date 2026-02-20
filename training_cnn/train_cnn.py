import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
import random
import numpy as np
import os
import sys

from cnn_model import ThreeBranchCNN
from dataset import NanoporeReadDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_experiment(window_size, cutoff):
    # Paths (derived from params)
    
    base_dir = "."
    data_dir = f"{base_dir}/test_data_window{window_size}_cutoff{cutoff}_test"

    human_dir = f"{data_dir}/human/"
    ecoli_dir = f"{data_dir}/ecoli/"

    print(f"Training on: {data_dir}")

    
    ds_human = NanoporeReadDataset(human_dir, label=0)
    ds_ecoli_full = NanoporeReadDataset(ecoli_dir, label=1)

    num_human = len(ds_human)
    print(f"Human reads: {num_human}")

    ecoli_indices = random.sample(range(len(ds_ecoli_full)), num_human)
    ds_ecoli = Subset(ds_ecoli_full, ecoli_indices)

    dataset = ConcatDataset([ds_human, ds_ecoli])

    BATCH_SIZE = 64
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ThreeBranchCNN(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total_samples = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total_samples += y.size(0)

        print(
            f"Epoch {epoch+1} | "
            f"loss: {total_loss / total_samples:.4f}, "
            f"acc: {correct / total_samples:.4f}"
        )


    # Save model
    experiment_name = os.path.basename(data_dir)
    model_name = f"{experiment_name}.pth"

    torch.save(model.state_dict(), model_name)
    print(f"Saved model: {model_name}")


if __name__ == "__main__":
    window_size = int(sys.argv[1])
    cutoff = float(sys.argv[2])
    train_experiment(window_size, cutoff)
