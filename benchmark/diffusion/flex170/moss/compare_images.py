"""Metrics in RGB8, including absolute error against each model's pre-change PNG."""

from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def pixels(p):
    with Image.open(p) as im:
        return np.array(im.convert("RGB"))


def metrics(x, y):
    delta = x.astype("int16") - y
    mse = float(np.square(delta.astype("float64")).mean())
    return dict(
        ssim=structural_similarity(x, y, channel_axis=2, data_range=255),
        psnr_db=float("inf") if not mse else 10 * np.log10(255**2 / mse),
        mae=np.abs(delta).mean(),
        max_abs=np.abs(delta).max(),
        changed_pixels=int(np.any(delta, axis=-1).sum()),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    print(metrics(pixels(args.image), pixels(args.reference)))
