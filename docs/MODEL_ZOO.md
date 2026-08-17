# Model Zoo

| Model | Visual prior | AP@0.25 | AP@0.50 | Weight |
|---|---|---:|---:|---|
| VPGNet | SAM3-FP1 | 0.9846 | 0.9794 | [Hugging Face](https://huggingface.co/alex135632/vpgnet-airport-luggage/blob/main/best.pth) |

Checkpoint metadata:

- Visual prior: SAM3-FP1
- Config: `configs/vpgnet/vpgnet_airport_luggage.py`
- SHA256: `0be40e94ab990aad9f1cd22b322fbc6957db7e52a8c9efb296e5044f5b52e1c5`
- Size: 186697088 bytes

Download page:

```text
https://huggingface.co/alex135632/vpgnet-airport-luggage/blob/main/best.pth
```

Download it to the expected local path with:

```bash
bash scripts/download_checkpoint.sh
```
