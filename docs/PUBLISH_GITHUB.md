# Publish to GitHub

This repository is prepared so the code can be pushed directly to GitHub.
Dataset files and model weights are excluded from git.

## Option A: Push with git

```bash
cd VPGNet-Airport-Luggage-OpenSource
git remote add origin git@github.com:<user>/<repo>.git
git push -u origin main
```

Then upload the checkpoint as a GitHub Release asset:

The local asset is in `../VPGNet-Airport-Luggage-ReleaseAssets/`.

## Option B: Upload the source zip

Upload this zip to GitHub or share it as the code package:

The local source zip is `../VPGNet-Airport-Luggage-OpenSource_github.zip`.

Upload the checkpoint separately as a Release asset:

The local asset is in `../VPGNet-Airport-Luggage-ReleaseAssets/`.
