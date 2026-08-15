import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model.model import RestorationModel


class SemiconductorDataset(Dataset):
    def __init__(self, noisy_dir, gt_dir):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.files = [
            f for f in os.listdir(noisy_dir)
            if f.endswith(".npy")
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.files[index]

        noisy = np.load(
            os.path.join(self.noisy_dir, filename)
        ).astype(np.float32)

        gt = np.load(
            os.path.join(self.gt_dir, filename)
        ).astype(np.float32)

        noisy = torch.from_numpy(noisy).unsqueeze(0)
        gt = torch.from_numpy(gt).unsqueeze(0)

        return noisy, gt


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NOISY_DIR = "data/train/NoisyLR"
GT_DIR = "data/train/GT"

dataset = SemiconductorDataset(
    NOISY_DIR,
    GT_DIR
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    num_workers=2
)

model = RestorationModel().to(device)

criterion = nn.L1Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 20

os.makedirs("weights", exist_ok=True)

best_loss = float("inf")

for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for noisy, gt in loader:

        noisy = noisy.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()

        output = model(noisy)

        loss = criterion(output, gt)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- Loss: {average_loss:.6f}"
    )

    if average_loss < best_loss:

        best_loss = average_loss

        torch.save(
            model.state_dict(),
            "weights/best_model.pth"
        )

        print("Best model saved.")
