# Model Zoo

| Model | Epoch | AP@0.25 | AP@0.50 | Weight |
|---|---:|---:|---:|---|
| VPGNet | 31 | 0.9803 | 0.9737 | `checkpoints/vpgnet_epoch31_best.pth` |

Checkpoint metadata:

- Config: `configs/vpgnet/vpgnet_airport_luggage.py`
- SHA256: `5f6fad2ed4949df44eaba6aaae90d3dfe97b0630dd16d7bc35c3da9ea52f8d34`
- Size: 212203916 bytes

Publishing note: the weight file is larger than 100 MiB and is not tracked in
the git repository. Upload it with Git LFS or as a GitHub Release asset instead
of a normal git blob.
