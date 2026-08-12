#!/usr/bin/env python3
"""Extract SAM3 FPN features for the VPGNet SUN RGB-D style dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


def install_torch_attention_compat() -> None:
    """Expose torch.nn.attention for SAM3 when running on older PyTorch."""
    if "torch.nn.attention" in sys.modules:
        return
    if hasattr(torch.nn, "attention"):
        return
    if not hasattr(torch.backends, "cuda"):
        return
    if not hasattr(torch.backends.cuda, "sdp_kernel"):
        return
    if not hasattr(torch.backends.cuda, "SDPBackend"):
        return

    module = types.ModuleType("torch.nn.attention")
    module.sdpa_kernel = torch.backends.cuda.sdp_kernel
    module.SDPBackend = torch.backends.cuda.SDPBackend
    sys.modules["torch.nn.attention"] = module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract SAM3 FPN features from sunrgbd_trainval/image and save "
            "them as .pt files readable by LoadSamFeature."
        ))
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/sunrgbd/sunrgbd_trainval/image"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sunrgbd/sunrgbd_trainval/sam_f"))
    parser.add_argument(
        "--sam3-repo",
        type=Path,
        default=Path(os.environ["SAM3_REPO"])
        if os.environ.get("SAM3_REPO") else None,
        help="Path to a local facebookresearch/sam3 checkout. Optional if "
        "the sam3 package is already importable.")
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=Path(os.environ["SAM_CHECKPOINT"])
        if os.environ.get("SAM_CHECKPOINT") else None,
        help="Path to sam3.pt. If omitted, SAM3 will try to download it from "
        "facebook/sam3 through Hugging Face.")
    parser.add_argument("--sam-resolution", type=int, default=1008)
    parser.add_argument(
        "--feature-level",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="SAM3 FPN level to save. The released VPGNet model uses FP1.")
    parser.add_argument(
        "--save-dtype",
        choices=("float16", "float32"),
        default="float16")
    parser.add_argument(
        "--save-format",
        choices=("tensor", "dict", "full"),
        default="tensor",
        help="'tensor' saves only the selected FPN tensor. 'dict' stores the "
        "selected tensor with metadata. 'full' stores fpn_0/fpn_1/fpn_2.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-from", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--empty-cache-every", type=int, default=50)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA autocast with float16.")
    return parser.parse_args()


def import_sam3(sam3_repo: Path | None):
    if sam3_repo is not None:
        sys.path.insert(0, str(sam3_repo.resolve()))

    install_torch_attention_compat()

    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except ImportError as exc:
        raise ImportError(
            "Could not import SAM3. Install facebookresearch/sam3 with "
            "`pip install -e /path/to/sam3` or pass --sam3-repo /path/to/sam3."
        ) from exc
    return build_sam3_image_model, Sam3Processor


def collect_images(image_dir: Path, start_from: str | None,
                   limit: int | None) -> list[Path]:
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if start_from is not None:
        image_paths = [p for p in image_paths if p.stem >= start_from]
    if limit is not None:
        image_paths = image_paths[:limit]
    return image_paths


def amp_context(enabled: bool, device: str):
    if enabled and device.startswith("cuda") and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def to_disk(tensor: torch.Tensor, save_dtype: str) -> torch.Tensor:
    dtype = torch.float16 if save_dtype == "float16" else torch.float32
    return tensor.detach().to(dtype=dtype).cpu().contiguous()


def atomic_save(payload: object, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def extract_one(image: Image.Image, image_path: Path, processor,
                feature_level: int, save_dtype: str, save_format: str,
                use_amp: bool, device: str) -> object:
    with torch.inference_mode(), amp_context(use_amp, device):
        state = processor.set_image(image)

    backbone_fpn = state["backbone_out"]["backbone_fpn"]
    if save_format == "full":
        return {
            "image_path": str(image_path),
            "original_size_wh": image.size,
            "processor_resolution": processor.resolution,
            "fpn_0": to_disk(backbone_fpn[0].squeeze(0), save_dtype),
            "fpn_1": to_disk(backbone_fpn[1].squeeze(0), save_dtype),
            "fpn_2": to_disk(backbone_fpn[2].squeeze(0), save_dtype),
        }

    key = f"fpn_{feature_level}"
    tensor = to_disk(backbone_fpn[feature_level].squeeze(0), save_dtype)
    if save_format == "dict":
        return {
            "image_path": str(image_path),
            "original_size_wh": image.size,
            "processor_resolution": processor.resolution,
            key: tensor,
        }
    return tensor


def main() -> None:
    args = parse_args()
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.sam_checkpoint is not None and not args.sam_checkpoint.is_file():
        raise FileNotFoundError(
            f"SAM3 checkpoint not found: {args.sam_checkpoint}")

    build_sam3_image_model, Sam3Processor = import_sam3(args.sam3_repo)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    image_paths = collect_images(args.image_dir, args.start_from, args.limit)
    if not image_paths:
        raise RuntimeError(f"No images found in {args.image_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_dir": str(args.image_dir),
        "output_dir": str(args.output_dir),
        "num_images": len(image_paths),
        "sam3_repo": str(args.sam3_repo) if args.sam3_repo else None,
        "sam_checkpoint": str(args.sam_checkpoint)
        if args.sam_checkpoint else None,
        "sam_resolution": args.sam_resolution,
        "feature_level": args.feature_level,
        "save_dtype": args.save_dtype,
        "save_format": args.save_format,
    }
    (args.output_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")

    print(f"Found {len(image_paths)} images in {args.image_dir}")
    print(f"Output dir: {args.output_dir}")
    if args.sam_checkpoint:
        print(f"Loading SAM3 checkpoint: {args.sam_checkpoint}")
    else:
        print("Loading SAM3 checkpoint from Hugging Face: facebook/sam3")

    model = build_sam3_image_model(
        checkpoint_path=str(args.sam_checkpoint)
        if args.sam_checkpoint else None,
        load_from_HF=args.sam_checkpoint is None,
        device=args.device,
        eval_mode=True)
    processor = Sam3Processor(
        model, resolution=args.sam_resolution, device=args.device)

    start_time = time.time()
    processed = 0
    skipped = 0
    failed = 0
    error_log = args.output_dir / "errors.log"

    for index, image_path in enumerate(image_paths, start=1):
        output_path = args.output_dir / f"{image_path.stem}.pt"
        if output_path.is_file() and not args.overwrite:
            skipped += 1
            continue

        image = Image.open(image_path).convert("RGB")
        try:
            payload = extract_one(
                image=image,
                image_path=image_path,
                processor=processor,
                feature_level=args.feature_level,
                save_dtype=args.save_dtype,
                save_format=args.save_format,
                use_amp=args.amp,
                device=args.device)
            atomic_save(payload, output_path)
            processed += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            with error_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{datetime.now().isoformat(timespec='seconds')} "
                    f"{image_path} {type(exc).__name__}: {exc}\n")
            print(f"Failed on {image_path.name}: {type(exc).__name__}: {exc}")
        finally:
            image.close()

        if args.empty_cache_every > 0 and index % args.empty_cache_every == 0:
            torch.cuda.empty_cache()
        if args.report_every > 0 and index % args.report_every == 0:
            elapsed = time.time() - start_time
            print(
                f"[{index}/{len(image_paths)}] processed={processed} "
                f"skipped={skipped} failed={failed} elapsed={elapsed:.1f}s")

    elapsed = time.time() - start_time
    print(
        f"Finished. processed={processed} skipped={skipped} failed={failed} "
        f"total_images={len(image_paths)} elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
