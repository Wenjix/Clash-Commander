"""
Clash Companion — Export card_resnet.pth to PyTorch Mobile Lite (.ptl)

Run on your laptop (4070) with PyTorch installed:
    pip install torch torchvision
    python scripts/export_card_resnet.py

This script:
  1. Downloads card_resnet.pth from GitHub (140KB)
  2. Inspects the state dict to determine architecture config
  3. Reconstructs the ResNet model class
  4. Loads weights and exports to .ptl (PyTorch Mobile Lite)
  5. Generates card_classes.json (class index → card name mapping)
  6. Optionally validates on local crop images if present

Output files (copy these to app/src/main/assets/):
  - card_resnet.ptl
  - card_classes.json
"""

import os
import sys
import json
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────

MODEL_URL = "https://raw.githubusercontent.com/shawnxu0407/Clash_Royale_agent/main/card_resnet.pth"
MODEL_PATH = "card_resnet.pth"
OUTPUT_PTL = "card_resnet.ptl"
OUTPUT_CLASSES = "card_classes.json"
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "src", "main", "assets")

# All 125 cards from katacr/constants/card_list.py (kebab-case)
ALL_CARDS = [
    "mirror",
    "electro-spirit", "fire-spirit", "heal-spirit", "ice-spirit",
    "ice-spirit-evolution", "skeletons", "skeletons-evolution",
    "barbarian-barrel", "bats", "bats-evolution", "bomber",
    "bomber-evolution", "giant-snowball", "goblins", "ice-golem",
    "rage", "spear-goblins", "the-log", "wall-breakers",
    "wall-breakers-evolution", "zap", "zap-evolution",
    "archers", "archers-evolution", "arrows", "bandit", "cannon",
    "clone", "dart-goblin", "earthquake", "elixir-golem",
    "firecracker", "firecracker-evolution", "fisherman",
    "goblin-barrel", "goblin-gang", "guards", "ice-wizard",
    "knight", "knight-evolution", "little-prince", "mega-minion",
    "miner", "minions", "princess", "royal-delivery", "royal-ghost",
    "skeleton-army", "skeleton-barrel", "tombstone", "tornado",
    "baby-dragon", "battle-healer", "battle-ram",
    "battle-ram-evolution", "bomb-tower", "dark-prince",
    "electro-wizard", "fireball", "flying-machine", "freeze",
    "furnace", "goblin-cage", "goblin-drill", "golden-knight",
    "hog-rider", "hunter", "inferno-dragon", "lumberjack",
    "magic-archer", "mighty-miner", "mini-pekka", "mortar",
    "mortar-evolution", "mother-witch", "musketeer", "night-witch",
    "phoenix", "poison", "skeleton-dragons", "skeleton-king",
    "tesla", "tesla-evolution", "valkyrie", "valkyrie-evolution",
    "zappies",
    "archer-queen", "balloon", "barbarians", "barbarians-evolution",
    "bowler", "cannon-cart", "electro-dragon", "executioner", "giant",
    "goblin-hut", "graveyard", "inferno-tower", "minion-horde",
    "monk", "prince", "ram-rider", "rascals", "royal-hogs", "witch",
    "wizard",
    "barbarian-hut", "elite-barbarians", "elixir-collector",
    "giant-skeleton", "goblin-giant", "lightning", "rocket",
    "royal-giant", "royal-giant-evolution", "sparky", "x-bow",
    "electro-giant", "lava-hound", "mega-knight", "pekka",
    "royal-recruits", "royal-recruits-evolution",
    "golem",
    "three-musketeers",
]


# ── Model Architecture (from torch_train.py) ──────────────────────────

@dataclass
class ModelConfig:
    filters: int = 4
    stage_sizes: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    num_class: int = 126


class BottleneckResNetBlock(nn.Module):
    def __init__(self, in_channels, filters, strides=1):
        super().__init__()
        out_channels = filters * 2

        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters, eps=1e-5, momentum=0.9)

        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=strides, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters, eps=1e-5, momentum=0.9)

        self.conv3 = nn.Conv2d(filters, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.9)
        nn.init.zeros_(self.bn3.weight)

        if strides != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, bias=False),
                nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.9)
            )
        else:
            self.proj = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        identity = self.proj(identity)
        return self.relu(out + identity)


class ResNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.conv1 = nn.Conv2d(1, cfg.filters, kernel_size=3, stride=2, padding=1, bias=False)
        self.act1 = nn.ReLU()

        self.stages = nn.ModuleList()
        in_channels = cfg.filters
        for i, num_blocks in enumerate(cfg.stage_sizes):
            blocks = []
            filters = cfg.filters * (2 ** i)
            for j in range(num_blocks):
                stride = 2 if j == 0 and i > 0 else 1
                blocks.append(BottleneckResNetBlock(
                    in_channels=in_channels,
                    filters=filters,
                    strides=stride
                ))
                in_channels = filters * 2
            self.stages.append(nn.Sequential(*blocks))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        final_out_channels = cfg.filters * (2 ** (len(cfg.stage_sizes) - 1)) * 2
        self.fc = nn.Linear(final_out_channels, cfg.num_class, bias=False)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ── Step 1: Download model ────────────────────────────────────────────

def download_model():
    if os.path.exists(MODEL_PATH):
        size = os.path.getsize(MODEL_PATH)
        print(f"[OK] {MODEL_PATH} already exists ({size:,} bytes)")
        return
    print(f"[GET] Downloading {MODEL_URL}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    size = os.path.getsize(MODEL_PATH)
    print(f"[OK] Downloaded {MODEL_PATH} ({size:,} bytes)")


# ── Step 2: Inspect state dict and determine config ───────────────────

def inspect_and_build_config(sd):
    print("\n-- State Dict Inspection --")

    # Determine filters from conv1
    conv1_shape = sd["conv1.weight"].shape
    filters = conv1_shape[0]
    print(f"  conv1.weight: {conv1_shape} -> filters = {filters}")

    # Determine stage_sizes by counting blocks per stage
    stage_blocks = {}
    for key in sd.keys():
        if key.startswith("stages."):
            parts = key.split(".")
            stage_idx = int(parts[1])
            block_idx = int(parts[2])
            stage_blocks.setdefault(stage_idx, set()).add(block_idx)

    stage_sizes = [len(blocks) for _, blocks in sorted(stage_blocks.items())]
    print(f"  stage_sizes = {stage_sizes}")

    # Determine num_class from fc layer
    fc_shape = sd["fc.weight"].shape
    num_class = fc_shape[0]
    print(f"  fc.weight: {fc_shape} -> num_class = {num_class}")

    # Total params
    total = sum(v.numel() for v in sd.values())
    print(f"  Total parameters: {total:,}")

    cfg = ModelConfig(filters=filters, stage_sizes=stage_sizes, num_class=num_class)
    print(f"\n  ModelConfig(filters={cfg.filters}, stage_sizes={cfg.stage_sizes}, num_class={cfg.num_class})")
    return cfg


# ── Step 3: Build class name mapping ──────────────────────────────────

def build_class_mapping(num_class):
    """
    Reconstruct the idx→card mapping the same way DatasetBuilder does:
    sorted alphabetically, then swap 'empty' to index 1.
    """
    # Add 'empty' to the card list
    cards_with_empty = ALL_CARDS + ["empty"]

    # Sort alphabetically (same as sorted(Path.glob('*')))
    cards_sorted = sorted(set(cards_with_empty))

    # DatasetBuilder swaps empty to index 1
    EMPTY_INDEX = 1
    if cards_sorted[EMPTY_INDEX] != "empty":
        natural_idx = cards_sorted.index("empty")
        cards_sorted[EMPTY_INDEX], cards_sorted[natural_idx] = (
            cards_sorted[natural_idx], cards_sorted[EMPTY_INDEX]
        )

    if len(cards_sorted) != num_class:
        print(f"\n  [WARN] Card list has {len(cards_sorted)} entries but model has {num_class} classes!")
        print(f"         The training dataset may have had a different card set.")
        print(f"         Class mapping may need manual correction after testing.")

        if len(cards_sorted) > num_class:
            cards_sorted = cards_sorted[:num_class]
        else:
            while len(cards_sorted) < num_class:
                cards_sorted.append(f"unknown_{len(cards_sorted)}")

    idx2card = {i: name for i, name in enumerate(cards_sorted)}

    print(f"\n-- Class Mapping ({len(idx2card)} classes) --")
    # Print first 10 and last 5 for verification
    for i in range(min(10, len(idx2card))):
        print(f"  {i}: {idx2card[i]}")
    if len(idx2card) > 15:
        print(f"  ...")
        for i in range(len(idx2card) - 5, len(idx2card)):
            print(f"  {i}: {idx2card[i]}")

    return idx2card


# ── Step 4: Export to PyTorch Mobile Lite ──────────────────────────────

def export_model(model, idx2card):
    model.eval()

    # Trace with dummy input: (batch=1, channels=1, height=80, width=64)
    dummy = torch.randn(1, 1, 80, 64)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    # Save for lite interpreter
    traced._save_for_lite_interpreter(OUTPUT_PTL)
    ptl_size = os.path.getsize(OUTPUT_PTL)
    print(f"\n[OK] Exported {OUTPUT_PTL} ({ptl_size:,} bytes)")

    # Save class mapping
    with open(OUTPUT_CLASSES, "w") as f:
        json.dump(idx2card, f, indent=2)
    print(f"[OK] Saved {OUTPUT_CLASSES} ({len(idx2card)} classes)")

    # Also copy to assets directory if it exists
    assets = os.path.abspath(ASSETS_DIR)
    if os.path.isdir(assets):
        import shutil
        shutil.copy2(OUTPUT_PTL, os.path.join(assets, OUTPUT_PTL))
        shutil.copy2(OUTPUT_CLASSES, os.path.join(assets, OUTPUT_CLASSES))
        print(f"[OK] Copied to {assets}")
    else:
        print(f"\n[INFO] Assets dir not found at {assets}")
        print(f"       Manually copy {OUTPUT_PTL} and {OUTPUT_CLASSES} to app/src/main/assets/")


# ── Step 5: Validate on local crops (optional) ────────────────────────

def validate_on_crops(model, idx2card):
    """Test the model on any crop_slot*.png or match_slot*.png files in project root."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..")

    crop_files = []
    for pattern in ["crop_slot*.png", "match_slot*.png"]:
        import glob
        crop_files.extend(glob.glob(os.path.join(project_root, pattern)))

    if not crop_files:
        print("\n[INFO] No crop images found for validation. Skipping.")
        return

    print(f"\n-- Validation on {len(crop_files)} local crops --")

    try:
        from PIL import Image
    except ImportError:
        print("[WARN] PIL not installed, skipping validation. pip install Pillow")
        return

    model.eval()
    for path in sorted(crop_files):
        img = Image.open(path).convert("L")  # grayscale
        img = img.resize((64, 80), Image.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 80, 64)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            top_prob, top_idx = torch.max(probs, dim=1)
            pred_name = idx2card.get(top_idx.item(), "???")

            # Top 3 predictions
            top3_probs, top3_idxs = torch.topk(probs, 3, dim=1)

        fname = os.path.basename(path)
        top3 = ", ".join(
            f"{idx2card.get(i.item(), '???')}({p.item():.2%})"
            for p, i in zip(top3_probs[0], top3_idxs[0])
        )
        print(f"  {fname}: {pred_name} ({top_prob.item():.2%}) | top3: {top3}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Clash Companion — ResNet Card Classifier Export")
    print("=" * 60)

    # Step 1: Download
    download_model()

    # Step 2: Inspect state dict
    sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    cfg = inspect_and_build_config(sd)

    # Step 3: Build class mapping
    idx2card = build_class_mapping(cfg.num_class)

    # Step 4: Reconstruct model and load weights
    print("\n-- Loading model --")
    model = ResNet(cfg)
    model.load_state_dict(sd)
    model.eval()
    print("[OK] Model loaded successfully")

    # Step 5: Export
    export_model(model, idx2card)

    # Step 6: Validate on local crops
    validate_on_crops(model, idx2card)

    print("\n" + "=" * 60)
    print("DONE. Next steps:")
    print("  1. Check the class mapping and validation results above")
    print(f"  2. Ensure {OUTPUT_PTL} and {OUTPUT_CLASSES} are in app/src/main/assets/")
    print("  3. Build and deploy: ./gradlew assembleDebug")
    print("=" * 60)


if __name__ == "__main__":
    main()
