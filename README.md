# Multiverse Origin Studio AI

A from-scratch manga/manhwa AI generation system — no pretrained backbone, single-file architecture.

---

## Overview

Multiverse Origin Studio AI is a full end-to-end pipeline for anime/manga-style image generation, covering dataset collection, training, and inference. The system is specifically designed to run on mid-range consumer hardware without compromising on output quality.

This is not a fine-tuned wrapper over Stable Diffusion or any existing model. Every component — architecture, loss functions, resource management, and data pipeline — is built from scratch and designed to work as a coherent chain. Each part of the system understands what the others need, so training quality scales with architectural depth rather than raw data volume.

**Verified result:** Successfully trained on 2,000+ curated images using 16GB RAM + RTX 3060. Estimated training time: 3–4 days continuous (varies by epoch count and dataset size). Quality-filtered data with domain-specific losses outperforms brute-force approaches requiring 10,000–50,000+ images.

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

**`EnhancedMoE` (14 Experts)** — Instead of a single monolithic model, tasks are routed to specialized expert agents. An intelligent router with learnable temperature selects and weights experts per input. Experts communicate through a cross-attention pass before their outputs are fused:

| Expert | Responsibility |
|--------|---------------|
| `line_art` | Edge structure, Laplacian-based thickness matching |
| `color` | Color palette generation, harmony, distribution matching |
| `light` | Lighting direction, intensity, shadow and specular generation |
| `anatomy` | Body proportion constraints, symmetry enforcement |
| `scene` | Scene layout, spatial composition |
| `camera` | Camera angle, projection matrix transformation |
| `story` | Multi-language narrative context (10 languages) |
| `structure` | High-level compositional structure |
| `precision` | Detail refinement and sharpness |
| `memory` | Temporal consistency across frames/panels |
| `safety` | Output constraint enforcement |
| `editing` | Region-based modification support |
| `quality` | Output quality scoring and enhancement |
| `style` | Style fingerprinting and artist-aware generation |

Each expert runs two passes — first independently, then again with cross-expert attention context — before outputs are fused through a multi-layer projection head.

**`EnhancedGNN`** — Graph Neural Network (v4, 6 layers, multi-head attention) connecting expert outputs as graph nodes. Enables relational reasoning across experts rather than treating their outputs as independent signals.

**`DataFlow`** — Thread-safe pipeline registry that handles module registration, inter-module connections, and ordered forward propagation. Allows dynamic rewiring of the computation graph without restructuring the model.

**`DiffusionModel`** — U-Net architecture with time embedding and cross-attention text conditioning.

**`VisionEncoder`** — Multi-scale CNN feature extraction: fine texture (3×3) → mid-level structure (pooled) → semantic summary (AdaptiveAvgPool). Returns separate embeddings for structure, color, and composition.

**`TextEncoder`** — Uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for real semantic embeddings. Falls back to a deterministic SHA-256 hash-based encoding (not random) if the model is unavailable, preserving reproducibility.

### Post-Processing Pipeline

After generation, outputs pass through a dedicated enhancement chain:

**`SuperResolutionSystem`** — 8-block residual SRGAN with PixelShuffle upsampling. Supports 2× and 4× upscaling with bicubic fallback for spatial alignment.

**`AdvancedDetailEnhancer`** — Multi-scale detail extraction (3×3 / 5×5 / 7×7 kernels) fused with dedicated edge and texture enhancement branches.

**`ColorVibrancySystem`** — Separate learnable adjustments for saturation, contrast, and brightness, followed by a color harmony pass.

**`MultiScaleProcessor`** — Processes image at 1×, 2×, and 4× downsampled scales simultaneously, upsamples back and fuses, preserving both global structure and fine local detail.

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

The system runs a multi-layer cache hierarchy specifically tuned for the R5 5500 + RTX 3060 combination, enabling stable multi-day training without manual intervention.

**`EnhancedResourceManager`** — Monitors RAM, VRAM, and CPU continuously. Computes dynamic batch size based on available resources and current resolution. Distinguishes between OS cache (not counted as used RAM) and actual process memory for accurate headroom estimation. Triggers graduated cleanup: soft at 90% RAM / 92% VRAM, aggressive at 95%.

**`L3CacheManager`** — Manages the CPU's 16MB L3 cache via an LRU eviction policy. Caches up to 1,500 tensor entries (~10KB each). Only caches tensors smaller than 50% of the cache budget to avoid thrashing. Tracks hit rate and eviction counts.

**`SSDOptimizer`** — Uses the NVMe SSD as a persistent tensor cache with a 50,000-entry index and a 256MB write buffer. Preprocessed image tensors are serialized as `.pt` files on first access and served directly from disk on all subsequent epochs, eliminating redundant CPU decode and transform overhead.

```python
# Resource allocation
max_ram_gb         = 14.4    # of 16GB  (1.6GB reserved for OS)
max_vram_gb        = 11.4    # of 12GB
cpu_threads        = 10      # R5 5500, 2 reserved for OS
ssd_cache_entries  = 50_000
ssd_buffer_size    = 256 * 1024 * 1024  # 256MB write buffer
l3_cache_max_mb    = 15      # of 16MB L3 (1MB reserved)
```

**Thread Management** — DataLoader runs 8 persistent workers (10 threads − 2 reserved). Uses `multiprocessing_context='spawn'` for Windows compatibility. `pin_memory=True` when CUDA is available for faster host-to-device transfers.

**Lazy Loading** — Dataset stores only file paths at initialization. Images are loaded on demand inside `__getitem__`, immediately serialized as `.pt` tensors to SSD cache, and served from cache on all subsequent accesses. Zero redundant image decoding after the first epoch.

**OOM Prevention** — Before any high-memory operation, `prevent_oom()` checks both RAM and VRAM headroom. On detection: triggers aggressive cleanup (double `gc.collect()` + `cuda.empty_cache()` + `cuda.synchronize()`), then re-checks before proceeding. Falls back to minimum batch size of 1 under critical conditions.

**Resolution-Aware Batch Sizing** — Batch size scales with target resolution: full batch at 512px, halved at 768px, quartered at 1024px, further reduced at 2048px. Prevents VRAM overflow without requiring manual tuning per run.

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
