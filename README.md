# Multiverse Origin Studio AI

A from-scratch manga/manhwa AI generation system — no pretrained backbone, single-file architecture.

---

## Overview

Multiverse Origin Studio AI is a full end-to-end pipeline for anime/manga-style image generation, covering dataset collection, training, and inference. The system is specifically designed to run on mid-range consumer hardware without compromising on output quality.

**Verified result:** Successfully trained on 2,000+ images using 16GB RAM + RTX 3060. Estimated training time: 3–4 days continuous (varies by epoch count and dataset size).

---

## Tested Hardware

| Component | Specification |
|-----------|--------------|
| CPU | AMD Ryzen 5 5500 (6 cores / 10 threads) |
| GPU | NVIDIA RTX 3060 (12GB VRAM) |
| RAM | 16 GB |
| Storage | M.2 NVMe SSD (used as cache layer) |

---

## Architecture

### Core AI Components

**`MultiverseOriginStudioAI`** — Top-level controller class. Manages training loop, generation, checkpointing, and resource orchestration.

**Mixture of Experts (MoE)** — Instead of a single monolithic model, tasks are delegated to specialized expert modules:

| Expert | Responsibility |
|--------|---------------|
| `LineArtExpert` | Edge matching, line thickness via Laplacian convolution |
| `ColorExpert` | Color theory, distribution matching, shading gradients |
| `BackgroundExpert` | Scene structure, perspective, low-frequency spatial content |
| `AdvancedCameraExpert` | Camera angles, projection matrix transformation |
| `MultiLanguageStorySystem` | Story generation across 10 languages (Thai, English, Japanese, Korean, Chinese, and more) |

**`DiffusionModel`** — U-Net architecture with time embedding and cross-attention text conditioning.

**`VisionEncoder`** — Multi-scale CNN feature extraction from fine texture details up to semantic-level understanding.

**`TextEncoder`** — Uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for real semantic embeddings. Gracefully falls back to hash-based encoding if the model is unavailable.

---

### Loss Functions

Each expert is trained with a domain-specific loss. Examples:

```python
# Line Art Loss — edge structure + thickness consistency
edge_matching  = F.l1_loss(line_edges, target_edges)
laplacian      = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]]).view(1,1,3,3)
thickness_loss = F.l1_loss(
    F.conv2d(line_art_gray, laplacian, padding=1),
    F.conv2d(target_gray,   laplacian, padding=1)
)
line_art_loss = 0.6 * edge_matching + 0.4 * thickness_loss

# Color Loss — distribution + variance + shading
color_dist_loss = F.mse_loss(gen_mean,  target_mean)
color_std_loss  = F.mse_loss(gen_std,   target_std)
shading_loss    = (F.l1_loss(grad_x, tgt_gx) + F.l1_loss(grad_y, tgt_gy)) / 2
color_loss = 0.4 * color_dist_loss + 0.3 * color_std_loss + 0.3 * shading_loss
```

Additional **Art Theory Losses** enforce higher-level artistic principles:
- **Anatomy** — bilateral symmetry check + edge proportion matching via Sobel filters
- **Perspective** — gradient magnitude consistency for depth cue alignment
- **Color Harmony** — penalizes over-saturated or clashing color distributions

---

## Resource Management

The system uses aggressive memory management to support multi-day continuous training.

**`EnhancedResourceManager`** — Monitors RAM, VRAM, and CPU in real time. Dynamically adjusts batch size and triggers cleanup before OOM conditions occur.

**`L3CacheManager`** — Maximizes use of the R5 5500's 16MB L3 cache via an LRU eviction policy. Supports up to 1,500 cached tensor entries.

**`SSDOptimizer`** — Uses the NVMe SSD as a persistent tensor cache. Preprocessed image tensors are written on first load and read directly on subsequent epochs, eliminating redundant disk I/O.

```python
# Resource allocation settings
max_ram_gb         = 14.4    # of 16GB  (1.6GB reserved for OS)
max_vram_gb        = 11.4    # of 12GB
ssd_cache_entries  = 50_000
ssd_buffer_size    = 256 * 1024 * 1024  # 256MB write buffer
```

**Lazy Loading** — The dataset stores only file paths at initialization. Images are loaded on demand inside `__getitem__`, immediately serialized as `.pt` tensors to SSD cache, and served from cache on all subsequent accesses.

---

## Dataset Collection

Multi-source downloader with strict content filtering:

| Source | Notes |
|--------|-------|
| Danbooru | Safe rating only + NSFW keyword blacklist (40+ terms) |
| Safebooru | Additional rating check |
| Konachan | Safe rating filter |
| Yande.re | Post-2015 only, score ≥ 3 |
| Wallhaven | Anime category, SFW purity flag |
| HuggingFace | `huggan/animeface-dataset` |
| Google CSE | Custom Search API with configurable query |

**Quality filters applied to every image:**
- Minimum resolution: 512px on the longer edge
- Aspect ratio: between 1:4 and 4:1
- Sharpness: Laplacian variance > 20 (accepts soft styles)
- Color variance: std > 10 (rejects near-blank images)
- Duplicate detection: MD5 hash checked against all previously downloaded images

---

## Training

```bash
# Basic training run
python ai_system_core.py --train --epochs 100 --batch_size 6

# Full configuration
python ai_system_core.py \
  --train \
  --data_dir ./data/train \
  --val_dir  ./data/val \
  --epochs   324 \
  --batch_size 2 \
  --save_path ./models/best_model.pt
```

**`AdaptiveTrainingSystem`** — Monitors convergence rate and loss variance to adjust the learning rate automatically during training. Reduces LR when the loss plateaus; increases it slightly when convergence is healthy.

**`AutomaticHyperparameterOptimizer`** — Bayesian-style search over learning rate, batch size, weight decay, and dropout. Runs configurable trial epochs before committing to a configuration.

Checkpoints are saved every 10 epochs and at milestones: 50, 100, 150, 200, 250, 300. The 20 most recent checkpoints are retained automatically.

---

## Inference

```bash
python ai_system_core.py \
  --generate \
  --model   ./models/best_model.pt \
  --prompt  "manga girl, forest background" \
  --output  result.png
```

---

## Visualization & Monitoring

```python
# Save a 3x3 training progress chart
visualize_training_progress(ai_system, save_path='progress.png')

# Print live dashboard to console
print_progress_dashboard(ai_system)
```

Tracked metrics: total loss, per-expert losses (line art / color / background), art theory losses (anatomy / perspective / color harmony), self-critique score, creativity score.

Logs are split by category for easier debugging:

| Log file | Content |
|----------|---------|
| `main.log` | General training events |
| `performance.log` | Per-batch timing and resource usage |
| `error.log` | Exceptions with context and recovery status |
| `diagnosis.log` | Automatic health checks per component |

---

## Project Structure

```
./
├── ai_system_core.py          # Full system — single file
├── data/
│   └── train/
│       ├── image001.jpg
│       ├── image001.txt       # Optional caption (paired by filename)
│       └── ...
├── models/
│   └── best_model.pt
├── cache/
│   └── preprocessed/          # Auto-generated SSD tensor cache
└── logs/
    ├── main.log
    ├── performance.log
    ├── error.log
    └── diagnosis.log
```

---

## Dependencies

```
torch
torchvision
Pillow
numpy
requests
sentence-transformers
transformers
psutil
matplotlib
seaborn
tensorboard
python-dotenv
tqdm
```

---

*Multiverse Origin Studio — All rights reserved*
