# Model Zoo

- We provide six pre-trained PointDiT models: PointDiT-B / L / H, each at `256x256` (Stage 1) and `512x512` (Stage 2). All weights are hosted on [Hugging Face](https://huggingface.co/haofeixu/pointdit).

- We assume that the downloaded weights are stored in the `pretrained` directory. It's recommended to create a symbolic link from `YOUR_MODEL_PATH` to `pretrained` using
```
ln -s YOUR_MODEL_PATH pretrained
```

- To verify the integrity of downloaded files, each model on this page includes its [sha256sum](https://sha256sum.com/) prefix in the file name, which can be checked using the command `sha256sum filename`.

- The `-nodinov3-` in every file name means the frozen DINOv3 encoder has been removed from the checkpoint: those weights are gated and cannot be redistributed here. They are downloaded once, separately, and re-attached at load time. See [DINOv3 encoder](#dinov3-encoder) below.


## PointDiT

- Stage 1 pre-trains at `256x256` on SceneNet-RGBD. Stage 2 fine-tunes that model at `512x512` on `mixdata`, an 11-dataset synthetic mixture ([dataloader/configs/res512mix.yaml](dataloader/configs/res512mix.yaml)).

- The header comment of each `scripts/eval_*.sh` lists the zero-shot numbers the corresponding model reproduces.

| Model                                        | Stage |      Training Data      |  Training Resolution   | DINOv3 Encoder |                                                    Download                                                    |
| -------------------------------------------- | :---: | :---------------------: | :--------------------: | :------------: | :------------------------------------------------------------------------------------------------------------: |
| pointditb-256-scenenet-nodinov3-58f0b231.pth |   1   |         scenenet        |        256x256         |    ViT-B/16    | [download](https://huggingface.co/haofeixu/pointdit/resolve/main/pointditb-256-scenenet-nodinov3-58f0b231.pth) |
| pointditl-256-scenenet-nodinov3-00141e97.pth |   1   |         scenenet        |        256x256         |    ViT-L/16    | [download](https://huggingface.co/haofeixu/pointdit/resolve/main/pointditl-256-scenenet-nodinov3-00141e97.pth) |
| pointdith-256-scenenet-nodinov3-9dddf3fc.pth |   1   |         scenenet        |        256x256         |   ViT-H+/16    | [download](https://huggingface.co/haofeixu/pointdit/resolve/main/pointdith-256-scenenet-nodinov3-9dddf3fc.pth) |
| pointditb-512-mixdata-nodinov3-1d42aacf.pth  |   2   | scenenet &rarr; mixdata | 256x256 &rarr; 512x512 |    ViT-B/16    | [download](https://huggingface.co/haofeixu/pointdit/resolve/main/pointditb-512-mixdata-nodinov3-1d42aacf.pth)  |
| pointditl-512-mixdata-nodinov3-240c1a4f.pth  |   2   | scenenet &rarr; mixdata | 256x256 &rarr; 512x512 |    ViT-L/16    | [download](https://huggingface.co/haofeixu/pointdit/resolve/main/pointditl-512-mixdata-nodinov3-240c1a4f.pth)  |
| pointdith-512-mixdata-nodinov3-cb01dd3b.pth  |   2   | scenenet &rarr; mixdata | 256x256 &rarr; 512x512 |   ViT-H+/16    | [download](https://huggingface.co/haofeixu/pointdit/resolve/main/pointdith-512-mixdata-nodinov3-cb01dd3b.pth)  |

The launch scripts in [scripts/](scripts) expect these files directly in `pretrained/`, under
exactly these names. They can also be fetched with the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"

# one model
hf download haofeixu/pointdit pointditl-512-mixdata-nodinov3-240c1a4f.pth --local-dir pretrained

# or all six (25 GB)
hf download haofeixu/pointdit --local-dir pretrained
```


## DINOv3 encoder

PointDiT conditions on a frozen DINOv3 encoder. Those weights are gated and cannot be
redistributed here, so they are not part of the checkpoints above and have to be downloaded
separately, once per model size.

Request access on the [official DINOv3 repository](https://github.com/facebookresearch/dinov3)
and place the weights in `pretrained/dinov3/`, under their exact upstream filenames:

```
pretrained/dinov3/
├── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth      # PointDiT-B
├── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth      # PointDiT-L
└── dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth  # PointDiT-H
```

If they live somewhere else, point `DINOV3_WEIGHTS_DIR` at that directory instead:

```bash
export DINOV3_WEIGHTS_DIR=/path/to/dinov3
```

Every launch script in [scripts/](scripts) reads that variable and falls back to
`pretrained/dinov3`. A run that finds a checkpoint but no encoder to pair it with stops with an
error rather than conditioning on a randomly initialised encoder.
