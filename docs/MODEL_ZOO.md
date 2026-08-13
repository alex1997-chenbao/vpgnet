# Model Zoo

| Model | Visual prior | AP@0.25 | AP@0.50 | Weight |
|---|---|---:|---:|---|
| VPGNet | SAM3-FP1 | 0.9869 | 0.9805 | [Hugging Face](https://huggingface.co/alex-chenbao1997/vpgnet-airport-luggage/blob/main/vpgnet_sam3fp1_best_epoch28.pth) |

Checkpoint metadata:

- Visual prior: SAM3-FP1
- Config: `configs/vpgnet/vpgnet_airport_luggage.py`
- SHA256: `150fa0fef23a6e1548a86511754a862cc22ac814e8b05dc6b524dce6ad4b87e1`
- Size: 188701348 bytes

Download it to the expected local path with:

```bash
bash scripts/download_checkpoint.sh
```
