# Model Zoo

Six checkpoints: each of PointDiT-B / L / H at 256x256 and 512x512 resolutions.

[Download the weights (Google Drive)](https://drive.google.com/drive/folders/1ctCGxi-WN1P5GpiteK41SXwLgQ1z5fEc?usp=drive_link) and symlink (or move) them under `pretrained/`:

```bash
ln -s YOUR_MODEL_PATH pretrained
```

The launch scripts expect the checkpoints directly in `pretrained/`, under these exact
filenames:

```
pretrained/
├── pointditb-256-scenenet.pth   # PointDiT-B, 256, Stage 1
├── pointditl-256-scenenet.pth   # PointDiT-L, 256, Stage 1
├── pointdith-256-scenenet.pth   # PointDiT-H, 256, Stage 1
├── pointditb-512.pth            # PointDiT-B, 512, Stage 2
├── pointditl-512.pth            # PointDiT-L, 512, Stage 2
└── pointdith-512.pth            # PointDiT-H, 512, Stage 2
```
