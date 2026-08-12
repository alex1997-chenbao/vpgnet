# Publish to GitHub

This repository is prepared so the code can be pushed directly to GitHub.
Dataset files and model weights are excluded from git.

## Option A: Push with git

```bash
cd VPGNet-Airport-Luggage-OpenSource
git push -u origin main
```

GitHub no longer accepts account passwords for HTTPS pushes. If this machine is
not logged in, use a GitHub personal access token as the password when prompted,
or install/authenticate GitHub CLI first.

Target repository:

```text
https://github.com/alex1997-chenbao/vpgnet
```

Companion dataset:

```text
https://huggingface.co/datasets/alex-chenbao1997/vpgnet-airport-luggage
```

Then upload the checkpoint as a GitHub Release asset:

```text
../VPGNet-Airport-Luggage-ReleaseAssets/vpgnet_sam3fp1_best_epoch28.pth
```

Checkpoint SHA256:

```text
150fa0fef23a6e1548a86511754a862cc22ac814e8b05dc6b524dce6ad4b87e1
```

## Option B: Upload the source zip

Upload this zip to GitHub or share it as the code package:

The local source zip is `../VPGNet-Airport-Luggage-OpenSource_github.zip`.

Upload the checkpoint separately as a Release asset:

```text
../VPGNet-Airport-Luggage-ReleaseAssets/vpgnet_sam3fp1_best_epoch28.pth
```
