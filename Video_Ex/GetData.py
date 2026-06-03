import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchrl.data.datasets import AtariDQNExperienceReplay

DATA_ROOT = r"..\data\Atari"
DATASET_IDS = [f"Breakout/{i}" for i in range(1, 6)]
FRAMES_PER_RUN = 20000
IMAGE_SIZE = 64


class AtariFrameDataset(Dataset):
    def __init__(self, frames, indices):
        self.frames = frames
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        storage, local_idx = self.frames[self.indices[idx]]
        sample = storage[local_idx]
        obs = sample.get("observation")
        if obs.ndim == 3:
            obs = obs.squeeze(0)
        x = obs.float().unsqueeze(0).unsqueeze(0) / 255.0
        x = F.interpolate(
            x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        )
        x = x.squeeze(0).repeat(3, 1, 1)
        return x, 0


def _collect_frames():
    frames = []
    for dataset_id in DATASET_IDS:
        ds = AtariDQNExperienceReplay(
            dataset_id,
            batch_size=1,
            root=DATA_ROOT,
            download=True,
        )
        picked = torch.randperm(len(ds))[:FRAMES_PER_RUN]
        storage = ds._storage
        for i in picked:
            frames.append((storage, int(i.item())))
    return frames


def get_AtariFrame(bs):
    frames = _collect_frames()
    perm = torch.randperm(len(frames))
    n_train = int(len(frames) * 0.9)
    train_idx = perm[:n_train].tolist()
    valid_idx = perm[n_train:].tolist()

    train_loader = DataLoader(AtariFrameDataset(frames, train_idx), bs, shuffle=True)
    valid_loader = DataLoader(AtariFrameDataset(frames, valid_idx), bs, shuffle=False)
    return train_loader, valid_loader
