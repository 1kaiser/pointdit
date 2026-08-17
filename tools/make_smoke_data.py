# coding=utf-8

"""Generate a tiny synthetic dataset so run.sh can exercise PointDiT offline.

This writes a handful of frames in the on-disk layout the Hypersim loader
expects (see ImageDepthIntrinsicsDataset._find_and_split_data_hypersim), plus a
couple of loose images for the wild-image inference path:

    <out>/data/<scene>/<subscene>/<frame:06d>_rgb.png     uint8 RGB
                                 /<frame:06d>_depth.npy   float32 z-depth, metres
                                 /<frame:06d>_cam.npz     key 'intrinsics', [3, 3]
    <out>/wild/<name>.png                                 uint8 RGB

The content is a smooth synthetic scene (a tilted ground plane with a box),
not random noise, so the depth is geometrically plausible and the point-map
conversion produces something meaningful to look at. Nothing here is training
data -- it exists purely so the smoke test needs no download.

Usage (from the repository root):
    python tools/make_smoke_data.py --out /tmp/pointdit_smoke
"""

import argparse
import os

import numpy as np
from PIL import Image


def _scene(height, width, seed):
    """Return (rgb uint8 [H, W, 3], depth float32 [H, W]) for one frame."""
    rng = np.random.RandomState(seed)
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    v = yy / max(height - 1, 1)
    u = xx / max(width - 1, 1)

    # A ground plane receding from the camera: depth grows towards the horizon.
    depth = 2.0 + 6.0 * (1.0 - v)
    # A nearer box occupying a rectangle in the middle of the frame.
    box = (v > 0.45) & (v < 0.8) & (u > 0.3) & (u < 0.7)
    depth[box] = 1.5 + 0.4 * u[box]
    # Mild per-frame variation so frames are not identical.
    depth = depth * (1.0 + 0.05 * rng.randn())
    depth = depth.astype(np.float32)

    # Shade the image from the depth so RGB and geometry are correlated.
    shade = (depth - depth.min()) / max(float(np.ptp(depth)), 1e-6)
    rgb = np.stack([1.0 - shade, 0.35 + 0.4 * u, shade], axis=-1)
    rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    return rgb, depth


def _intrinsics(height, width):
    """A plausible pinhole K for a ~60 degree horizontal field of view."""
    focal = 0.5 * width / np.tan(np.deg2rad(60.0) / 2.0)
    return np.array([
        [focal, 0.0, width / 2.0],
        [0.0, focal, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description='Write a tiny synthetic dataset for the PointDiT smoke test')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--size', type=int, default=96,
                        help='Square image side in pixels')
    parser.add_argument('--scenes', type=int, default=2)
    parser.add_argument('--frames', type=int, default=2,
                        help='Frames per scene')
    parser.add_argument('--wild', type=int, default=2,
                        help='Loose images for the --eval_wild_images path')
    args = parser.parse_args()

    size = args.size
    K = _intrinsics(size, size)

    seed = 0
    for scene in range(args.scenes):
        subscene_dir = os.path.join(
            args.out, 'data', f'ai_{scene:03d}_001', 'cam_00')
        os.makedirs(subscene_dir, exist_ok=True)
        for frame in range(args.frames):
            rgb, depth = _scene(size, size, seed)
            seed += 1
            stem = os.path.join(subscene_dir, f'{frame:06d}')
            Image.fromarray(rgb).save(f'{stem}_rgb.png')
            np.save(f'{stem}_depth.npy', depth)
            np.savez(f'{stem}_cam.npz', intrinsics=K)

    wild_dir = os.path.join(args.out, 'wild')
    os.makedirs(wild_dir, exist_ok=True)
    for i in range(args.wild):
        rgb, _ = _scene(size, size, 1000 + i)
        Image.fromarray(rgb).save(os.path.join(wild_dir, f'wild_{i:02d}.png'))

    n_train = args.scenes * args.frames
    print(f'Wrote {n_train} training frame(s) under {args.out}/data '
          f'and {args.wild} image(s) under {args.out}/wild '
          f'({size}x{size}).')


if __name__ == '__main__':
    main()
