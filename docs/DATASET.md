# Dataset Notes

The dataset is not included in this package.

The released config expects a SUN RGB-D style root:

```text
data/sunrgbd/
  points/
  sunrgbd_trainval/
    image/
    sam_f/
    sunrgbd_infos_train.pkl
    sunrgbd_infos_val.pkl
```

The dataset class is `LuggageSUNRGBDDataset`. It uses one class:

```text
luggage
```

The default pipeline loads:

- point cloud files from `points/`
- images from `sunrgbd_trainval/image/`
- SAM3 features from `sunrgbd_trainval/sam_f/`
- annotations from `sunrgbd_trainval/sunrgbd_infos_train.pkl` and
  `sunrgbd_trainval/sunrgbd_infos_val.pkl`

SAM feature files should be `.pt` tensors or dict payloads containing one of
these tensor keys: `sam_feat`, `patch_features`, `feature`, `features`,
`fpn_0`, `fpn_1`, or `fpn_2`.

