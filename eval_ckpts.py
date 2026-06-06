"""Throwaway: recover a comparable FCMAE recon-loss curve from saved checkpoints.

Each checkpoint is scored on the SAME fixed batch with the SAME mask (seed reset
before every forward), so the only thing varying is the model weights.
"""
import torch
from torch.utils.data import DataLoader

from src.core.utils import seed_everything, get_device, DATA_DIR, RAND_SEED
from src.core.dataset import PretrainDataset, TrainLabeledDataset, TrainMaskedDataset
from src.runners.pretrain import build_transform, build_model
from src.core.utils import PretrainConfigs
import dataclasses

cfg = dataclasses.replace(PretrainConfigs(), model_size="atto", batch_size=64, limit=1000)
device = get_device()
seed_everything(RAND_SEED)

tf = build_transform(cfg)
ds = PretrainDataset(DATA_DIR, tf, [TrainLabeledDataset(DATA_DIR, tf), TrainMaskedDataset(DATA_DIR, tf)])
ds.items = ds.items[:cfg.limit]
batch = next(iter(DataLoader(ds, batch_size=64, shuffle=True, num_workers=0))).to(device)
print(f"eval batch: {tuple(batch.shape)} from pool of {len(ds)}")

model = build_model(cfg).to(device).eval()
print(f"{'ckpt':<16}{'recon_loss':>12}")
for e in range(5):
    sd = torch.load(f"checkpoints/checkpoint-{e}.pth", map_location=device, weights_only=False)["model"]
    model.load_state_dict(sd, strict=True)
    with torch.no_grad():
        torch.manual_seed(0)  # identical mask every time
        loss, _, _ = model(batch, mask_ratio=cfg.mask_ratio)
    print(f"checkpoint-{e}.pth {float(loss):>11.4f}")
