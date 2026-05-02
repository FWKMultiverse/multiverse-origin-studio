# Multiverse Origin Studio AI

> **All rights reserved — Multiverse Origin Studio**
> Built from scratch by a solo developer. No pretrained image generation backbone.
> Single-file architecture (~10,000 lines). Verified training on consumer hardware.

---

**Multiverse Origin Studio AI is a complete end-to-end anime and manga image generation pipeline — architecture, loss functions, data pipeline, and resource management built entirely from scratch — designed to produce high-quality output on hardware that existing systems treat as insufficient.**

This is not a wrapper, fine-tune, or derivative of Stable Diffusion, Midjourney, or any existing image generation model. Every component was designed with awareness of what every other component requires.

**Verified result:** Successfully trained on 3,500+ curated images using 16GB RAM + RTX 3060 12GB VRAM. Estimated training time: 3–4 days continuous. Quality-filtered data with domain-specific losses outperforms brute-force approaches requiring 10,000–50,000+ images.

---

## Tested Hardware

| Component | Specification |
|-----------|--------------|
| CPU | AMD Ryzen 5 5500 — 6 cores / 10 threads, 16MB L3 cache |
| GPU | NVIDIA RTX 3060 — 12GB VRAM |
| RAM | 16 GB |
| Storage | M.2 NVMe SSD — persistent preprocessed tensor cache |

Every allocation limit, cache budget, and batch scaling threshold references these numbers directly. This is not a generic configuration — it was derived from measured hardware capacity.

---

## Architecture Overview

### System Controller

**`MultiverseOriginStudioAI`** — Top-level orchestrator. Initializes, wires, and coordinates all subsystems. Device placement is explicit and intentional:

**GPU** (quality-critical, performance-sensitive):
- `EnhancedGNN` v1–v4, `EnhancedMoE` (14 experts)
- `EmotionalArtisticUnderstanding`, `DeepUnderstandingSystem`
- `DiffusionModel`, `VisionEncoder`
- `AdvancedColorExpert`, `AdvancedLineArtExpert`, `BackgroundExpert`
- `AdvancedCameraExpert`, `StructureGenerator`, `AdvancedEditingExpert`

**CPU** (VRAM-constrained, deferred):
- `QualityEnhancer` — final post-processing stage, runs CPU-side to preserve VRAM throughout generation
- `AdvancedCompressionSystem` — embedding compression, optional
- `DesignStyleSupportSystem` — style conditioning, CPU-placed

**Dimension bridge:** `nn.Linear(1024→512) → LayerNorm → ReLU` — bridges MoE output to the 512-dim internal modules without restructuring any model.

---

### Mixture of Experts — 14 Specialists

**`EnhancedMoE`** routes each generation task through 14 specialized expert agents rather than a single monolithic model.

**Router architecture:**
```
Linear(768→1024) → LayerNorm → GELU → Dropout(0.1)
  → Linear(1024→512) → LayerNorm → GELU
  → Linear(512→14)
  → softmax(logits / temperature)
```
`temperature` is a learned `nn.Parameter` — the router learns how confident to be in its routing decisions, not just which expert to select.

**Each expert (`EnhancedExpertAgent`) — full architecture:**
```
input_proj: Linear(768→512)

layer1: LayerNorm → Linear(512→512) → GELU → Dropout(0.1) + residual from input_proj
layer2: LayerNorm → Linear(512→512) → GELU → Dropout(0.1) + residual from layer1

# Context integration (when context tensor provided):
attention: MultiheadAttention(512, heads=8, batch_first=True, dropout=0.1)
  h = h + attention(h, context, context).squeeze(1) × 0.5

comm_layer: Linear(512→512) → LayerNorm → GELU, added × 0.3

output_proj: Linear(512→512) × expert_weights (nn.Parameter)
```

Per-expert performance tracking — `call_count`, `avg_time` (EMA: α=0.1), `success_rate` (EMA: α=0.05/0.1) — logged live.

**14 expert domains:**

| Expert | Domain |
|--------|--------|
| `line_art` | Edge structure, stroke continuity, Laplacian thickness matching |
| `color` | Palette generation, color harmony, distribution and shading |
| `light` | Lighting direction, intensity, shadow and specular modeling |
| `anatomy` | Body proportion, bilateral symmetry enforcement |
| `scene` | Spatial composition, layout reasoning |
| `camera` | Camera angle, projection matrix, perspective transformation |
| `story` | Multi-language narrative conditioning (10 languages via `MultiLanguageStorySystem`) |
| `structure` | High-level compositional structure and form |
| `precision` | Detail refinement, local sharpness |
| `memory` | Temporal consistency across panels and sequences |
| `safety` | Output constraint enforcement |
| `editing` | Region-based modification and inpainting |
| `quality` | Output quality scoring and per-sample enhancement |
| `style` | Style fingerprinting, artist-aware generation |

**Two-pass execution — how experts actually communicate:**

Pass 1: All 14 experts run independently on input `x`, producing `expert_hiddens[]`.

Pass 2: Expert outputs are stacked into `[batch, 14, 512]` and passed through:
```
cross_attention: MultiheadAttention(512, heads=16, batch_first=True, dropout=0.1)
expert_tensor = expert_tensor + attended × 0.5   # cross-expert residual

coordinator: Linear(512→512) → LayerNorm → GELU
expert_tensor = expert_tensor + coordinated × 0.3

# Weighted combination
weighted_outputs = (expert_tensor × routing_weights.unsqueeze(-1)).sum(dim=1)

# Fusion head
fusion: Linear(512×14 → 512×4) → LayerNorm → GELU → Dropout(0.1)
     → Linear(512×4 → 512×2) → LayerNorm → GELU → Dropout(0.1)
     → Linear(512×2 → 512)

# Final
final_output = weighted_outputs + fused × 0.5
```

Routing history tracked in `deque(maxlen=1000)` per forward call.

---

### Graph Neural Network — Four Versions

**`EnhancedGNN`** — 6-layer network connecting expert outputs as graph nodes. Enables relational reasoning between experts: the output of one expert can attend to and modify the representation of another through message passing.

**Architecture (version 4 — active):**
```
node_embedding: Linear(768→512)

Per layer (v4):
  MultiheadAttention(512, heads=8, batch_first=True)
  LayerNorm + residual (from v3+)
  ReLU → Dropout(0.1)

output_proj: Linear(512→512)
```

- v4: MultiheadAttention per layer + residual + LayerNorm
- v3: Linear per layer + residual + LayerNorm
- v1–v2: Linear per layer + LayerNorm, no residual

All four versions are initialized and live. `gnn_v4` is registered in `DataFlow` as the active graph module. `gnn_v1`–`gnn_v3` are retained for ablation and fallback. `DataFlow.connect('gnn', 'graph_features', 'structure', 'embedding')` wires graph features directly into structure generation.

---

### Computation Graph — DataFlow

**`DataFlow`** is a thread-safe pipeline registry backed by `threading.Lock()`. Modules declare their names, input keys, and output keys. `DataFlow.forward()` resolves and executes them in declared order, passing results through the dependency graph.

Module registration:
```python
data_flow.register_module('line_art', line_art_expert,
    inputs=['embedding', 'structure'],
    outputs=['line_art'])
data_flow.connect('gnn', 'graph_features', 'structure', 'embedding')
data_flow.connect('structure', 'structure', 'line_art', 'structure')
data_flow.connect('structure', 'structure', 'background', 'structure')
```

New processing stages can be inserted by registration alone — no existing module code is modified.

---

### Diffusion Model

**`DiffusionModel`** — U-Net with sinusoidal time embedding and cross-attention text conditioning (768-dim). Text embeddings are injected at multiple scales via cross-attention, influencing both structure and fine detail rather than only at the bottleneck.

---

### Encoders

**`TextEncoder`** — `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim output). In `ArtDataset.__getitem__`, the 384-dim output is duplicated to 768 via `torch.cat([text_embed, text_embed])`. Falls back to a deterministic SHA-256 hash embedding (not random) when the transformer is unavailable — `_create_hash_embedding` maps 96 bytes of the hash to `[-1, 1]` bit-level values and fills remaining dimensions from character frequency statistics. Reproducibility is preserved without the transformer dependency.

**`VisionEncoder`** — 4-layer CNN backbone → `AdaptiveAvgPool2d(1)` → 512-dim → 4 separate output heads:
- `embedding`: `Linear(512→768) → LayerNorm → ReLU → Linear(768→768)`
- `structure`: `Linear(512→256) → ReLU → Linear(256→128)`
- `color`: `Linear(512→256) → ReLU → Linear(256→128)`
- `composition`: `Linear(512→256) → ReLU → Linear(256→128)`

**Important:** `VisionEncoder` is not used inside `ArtDataset.__getitem__` — this was an explicit design decision to avoid device mismatch between DataLoader worker processes and the GPU. Instead, `_create_image_stat_embedding` always generates exactly 768 dimensions from per-channel statistics (mean, std, min, max), 50-bin histograms per channel, and cross-channel correlations. `VisionEncoder` is used during inference only, when device context is controlled.

---

### Emotional and Artistic Understanding

**`EmotionalArtisticUnderstanding`** — GPU-placed system conditioning generation on emotional and artistic analysis. Architecture:

```
emotion_encoder:
  Linear(768→1536) → LayerNorm → GELU → Dropout(0.1)
  → Linear(1536→768) → LayerNorm

emotion_classifier:
  Linear(768→512) → ReLU → Linear(512→256) → ReLU → Linear(256→10) → Softmax
```

10 emotion categories: joy, sadness, anger, fear, surprise, disgust, neutral, love, nostalgia, melancholy.

Five artistic element extractors (`composition`, `color_harmony`, `lighting_mood`, `perspective_emotion`, `line_quality`), each `Linear(768→256) → ReLU → Linear(256→128)`.

```
artistic_style_encoder:
  Linear(768→1536) → LayerNorm → GELU → Dropout(0.1) → Linear(1536→768)

artistic_fusion:
  Linear(768 + 128×5 → 1536) → LayerNorm → GELU → Dropout(0.1) → Linear(1536→768)

emotion_to_visual bridge:
  Linear(768+10 → 1536) → LayerNorm → GELU → Linear(1536→768)
```

Emotion scores (10-dim) are concatenated directly to the generation embedding and passed through the bridge — emotional analysis directly modulates the visual representation, not just a soft conditioning signal.

---

### Deep Understanding System

**`DeepUnderstandingSystem`** — GPU-placed multi-level understanding stack:

```
semantic_understanding:
  Linear(768→1536) → LayerNorm → GELU → Dropout(0.1) → Linear(1536→768)

contextual_encoder:
  TransformerEncoder(d_model=768, nhead=12, dim_ff=3072, layers=6,
                     batch_first=True, dropout=0.1)
  → mean over sequence

hierarchical_layers: 4 × [Linear(768→768) → LayerNorm → GELU]

relationship_net:
  Linear(1536→1536) → LayerNorm → GELU → Linear(1536→768)

understanding_fusion:
  Linear(768×3 → 1536) → LayerNorm → GELU → Dropout(0.1) → Linear(1536→768)

attention: MultiheadAttention(768, heads=12, batch_first=True)
```

Semantic, contextual (TransformerEncoder), and hierarchical representations are fused together before generation.

---

### Advanced Compression System

**`AdvancedCompressionSystem`** — CPU-placed. Compresses 768-dim embeddings at `compression_ratio=0.5` → 384-dim for VRAM-critical paths.

```
importance_net:  Linear(768→384) → ReLU → Linear(384→768) → Sigmoid
  (importance scores [0,1] weight embedding before compression)

compressor:
  Linear(768→1536) → LayerNorm → GELU → Linear(1536→768) → LayerNorm → GELU → Linear(768→384)

decompressor:
  Linear(384→768) → LayerNorm → GELU → Linear(768→1536) → LayerNorm → GELU → Linear(1536→768)
```

8-bit quantization applied at inference (`scale = abs_max / 127.0; quantized = round(compressed / scale) * scale`). Reconstruction loss is `F.mse_loss(decompressed, original)` for training.

---

### Post-Processing Pipeline

**`QualityEnhancer`** — CPU-placed final stage. Orchestrates the full post-processing chain with 4 modes: `full`, `resolution`, `detail`, `color`. In `full` mode: MultiScaleProcessor → AdvancedDetailEnhancer → ColorVibrancySystem → base enhancer → final blend `enhanced + 0.3×details + 0.2×vibrant`. Staged quality enhancement (`_apply_staged_quality_enhance`) downscales to `max_side` (VRAM-based: 320–896px) before processing and upscales back — VRAM-neutral even for large outputs.

**`SuperResolutionSystem`** — 8-block residual SRGAN with PixelShuffle upsampling:
```
feature_extractor: Conv2d(3→64) → PReLU → Conv2d(64→64) → PReLU
residual_blocks (×8): Conv2d(64→64) → BN → PReLU → Conv2d(64→64) → BN + residual
upsampler: Conv2d(64→256) → PixelShuffle(2) → PReLU  [+ second stage for 4×]
reconstructor: Conv2d(64→64) → PReLU → Conv2d(64→3)
```
Bicubic fallback applied if spatial dimensions don't match target after PixelShuffle.

**`AdvancedDetailEnhancer`** — Three parallel detail branches (3×3 fine / 5×5 medium / 7×7 coarse), each `Conv2d(3→128) → BN → ReLU → Conv2d(128→128)`. Concatenated → `detail_fusion (128×3→256→256→128)`. Added to `edge_enhancer` (3×3) and `texture_enhancer` (3×3) outputs: `combined = image + 0.3×fused_mean + 0.2×edge + 0.2×texture`. Final `final_enhancer (3→256→256→128→3)` added residually × 0.4.

**`ColorVibrancySystem`** — Color analysis CNN (3→128→256→128) feeding three separate Tanh-activated branches for saturation (× `vibrancy_strength=0.3`), contrast (× 0.5×strength), and brightness (× 0.3×strength). Branches are decoupled to prevent coupling artifacts. Color harmony CNN (3→128→256→3) applied post-adjustment. Final `color_enhancement (3→256→256→128→3)` added × 0.4.

**`MultiScaleProcessor`** — Three scale branches at original / ½ / ¼ resolution, each with different kernel sizes (3×3 / 5×5 / 7×7). Upsampled to original → concatenated → `scale_fusion (128×3→256→256→128)` → `output_generator (128→256→256→3)`. Residual connection: `image + 0.5×output`.

---

## Loss Functions

Every expert is trained with a loss matched to its domain constraint — not a generic pixel reconstruction objective applied uniformly.

**`AdvancedLossFunctions`** assembles the full loss stack.

**VGG Perceptual Loss** — VGG feature-space loss. Produces sharper, more visually coherent output than pixel MSE by matching perceptual features (texture, style, high-level structure).

**Combined Loss (training path)** — Used for MoE and generator joint training:
- Mixed precision path: pure L1 loss (scale-stabilized)
- Standard precision path: `0.5×MSE + 0.5×L1` (configurable via `weights` dict)

**GAN Loss** — BCE loss with label smoothing: `real_label=0.9`, `fake_label=0.1`. Generator optimizer: AdamW `lr=5e-6`, `betas=(0.9, 0.999)`, AMSGrad. Discriminator optimizer: AdamW `lr=2.5e-6` (half generator LR), `betas=(0.5, 0.999)`, AMSGrad.

**SSIM Loss:**
```python
ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) /
           ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
loss = 1 - ssim_map.mean()
```

**Anatomical Loss (`AnatomicalDetector`)** — Three region-specific CNN detectors:
- Face: `Conv2d(3→64, k=7, stride=2) → BN → ReLU → MaxPool → Conv2d(64→128) → BN → ReLU → Conv2d(128→256) → BN → ReLU → AdaptiveAvgPool(1,1)`
- Hands: `Conv2d(3→64, k=5, stride=2) → BN → ReLU → Conv2d(64→128) → BN → ReLU → AdaptiveAvgPool(1,1)`
- Eyes: `Conv2d(3→32) → ReLU → Conv2d(32→64) → ReLU → AdaptiveAvgPool(1,1)`

Loss weighting: `0.5×face + 0.3×hands + 0.2×eyes`

**Style Consistency Loss** — Gram matrix matching between generated and target style embeddings. Weight: 0.2 when style loss is enabled.

**Temporal Consistency Loss (panel generation mode):**
```
0.5×feature_consistency + character_weight × character_consistency
                        + scene_weight × scene_consistency
```
Outpainting overlap loss: 32-pixel border zones on all 4 sides, MSE between generated border and reference border. Attention-based edge loss: Sobel edge magnitude map of reference used as attention weight for generated edge alignment. Combined: `loss + 0.2×overlap_loss + 0.1×attention_edge_loss`.

**`AnimeStyleDiscriminator`** — CNN trained for anime/manga aesthetics specifically:
```
Conv2d(3→48, k=3, stride=2) → LeakyReLU(0.2) → Dropout2d(0.25)
Conv2d(48→96, k=3, stride=2) → BN → LeakyReLU(0.2) → Dropout2d(0.25)
Conv2d(96→192, k=3, stride=2) → BN → LeakyReLU(0.2)
Conv2d(192→384, k=3, stride=2) → BN → LeakyReLU(0.2)
anime_features: Conv2d(384→192, k=1) → LeakyReLU → Conv2d(192→96, k=1) → LeakyReLU
classifier: AdaptiveAvgPool2d(1) → Flatten → Linear(96→1) → Sigmoid
```
Base channels = 48. Prevents drift toward photorealistic features.

---

## Dataset Pipeline

**`ArtDataset`** — Lazy loading: `__init__` stores file paths only. On first access per sample, image is decoded, transformed, and saved to `./cache/preprocessed/<name>.pt`. All subsequent accesses load the `.pt` directly — no image decoding or transform overhead after first epoch. `torch.save` / `torch.load(weights_only=False)` with corrupted-cache fallback.

**Text embedding:** `sentence-transformers` → 384-dim → duplicated to 768. Fallback: `_create_hash_embedding` (SHA-256 → 768-dim deterministic, not random).

**Image embedding inside DataLoader workers:** Always `_create_image_stat_embedding` (768-dim from channel statistics + histograms + correlations) — never `VisionEncoder` inside workers, to avoid device mismatch. `VisionEncoder` is used only in the `generate()` path when device context is controlled.

**`smart_collate_fn`** — Handles variable-length embeddings with zero-padding to batch max. Stacks style fingerprints separately when style extraction is enabled.

**Style pipeline:** `StyleAnalyzer` runs on each sample when `extract_style=True`, keyed by folder name or filename prefix into `ArtistStyleDatabase`.

**`_create_dummy_samples` is disabled** — raises `ValueError` with instructions rather than generating synthetic data. Real images are required.

Supported formats: PNG, JPG, JPEG, WEBP, BMP.

---

## Quality Filters (Image Download)

Applied to every image in `download_training_images`. Target: 3,500 images. Parallel hash computation via `ThreadPoolExecutor(max_workers=8)`.

| Filter | Threshold |
|--------|----------|
| Minimum long edge | 512px |
| Minimum short edge | 256px |
| Aspect ratio | 0.33–3.0 (stricter than 0.25–4.0 to exclude ultra-wide) |
| Color std | > 15 (raised from 10) |
| Sharpness (Laplacian variance) | > 30 (raised from 20) |
| Contrast (pixel std) | > 20 |
| Unique color check | > 100 unique colors if near pure black/white |
| Duplicate detection | Full-file MD5 hash |

NSFW: 40+ keyword blacklist applied to tag strings. Anime validation: tag string must contain at least one of `anime`, `manga`, `cartoon`, `illustration`, `drawing`, `art`, `character`. Danbooru score filter: ≥ 5.

If dataset already meets target → skip. If over target → deterministic trim (sorted reverse, remove last).

---

## Resource Management

**`EnhancedResourceManager`** — Monitors RAM, VRAM, CPU via `psutil` and `torch.cuda`. Distinguishes OS page cache from process memory. Computes dynamic batch size from live available resources and current resolution.

**Hard allocation budget:**
```
max_ram_gb     = 14.4   # 16GB − 1.6GB OS reserve
max_vram_gb    = 11.4   # 12GB − 0.6GB reserve
cpu_threads    = 10     # 2 reserved for OS and background
available_threads = 8   # used for DataLoader workers
```

**Cleanup thresholds:**
```
RAM:   soft=90%   aggressive=95%
VRAM:  soft=92%   aggressive=95%
Aggressive: gc.collect() × 2 + cuda.empty_cache() + cuda.synchronize()
```

**`L3CacheManager`** — LRU eviction over CPU L3 via `OrderedDict`. Budget: 15MB (1MB reserved from 16MB). Max entries: 1,500 (~10MB each). Tensors larger than 50% of budget are not cached (anti-thrash). Tracks hit rate, miss rate, eviction count. Used at inference time to cache `gnn_edge_index` and GNN outputs (< 1MB threshold).

**`SSDOptimizer`** — NVMe read cache. First access: decode → transform → `torch.save('./cache/preprocessed/<name>.pt')`. Subsequent accesses: `torch.load()` direct. Eliminates all image decoding and transform overhead after first epoch.

**DataLoader:** 8 workers (`num_workers = min(8, available_threads)`), `multiprocessing_context='spawn'`, `pin_memory=True` when CUDA available, `persistent_workers=True`, `prefetch_factor=2` when RAM > 2GB.

**Resolution-aware batch scaling:**
```
512px  → full batch
768px  → batch // 2
1024px → batch // 4
2048px → batch // 8
minimum: 1
```

**Dynamic resolution selection** (`get_optimal_resolution`) — based on live `vram_available_gb`:
```
> 8GB  → 2048px max
> 5GB  → 1024px max
> 2GB  → 768px max
else   → 512px
```
Further constrained if `ram_available_gb < 4`.

**OOM recovery:** On `RuntimeError` containing `out of memory` or `oom`:
1. Save emergency checkpoint to `./checkpoints/emergency_oom_epoch_<N>.pt`
2. Reduce batch size by 1 and recreate DataLoader
3. Recompute `grad_accum_steps` based on available VRAM (2/3/4)
4. If already at batch size 2, stop training

---

## Training System

**`TrainingSystem`** — Full training orchestration.

**Optimizer:** AdamW, `lr=5e-6`, `betas=(0.9, 0.999)`, `weight_decay=1e-5`, `eps=1e-8`, `amsgrad=True`. Parameters: MoE + `structure_generator` + `line_art_expert` + `background_expert` + `color_expert` + `camera_expert` + `story_expert`.

**Discriminator optimizer:** AdamW, `lr=2.5e-6` (half generator LR), `betas=(0.5, 0.999)`, `amsgrad=True`.

**Scheduler:** `CosineAnnealingWarmRestarts(T_0=65, T_mult=2, eta_min=1e-8)`. Warmup: 44 epochs (~17% of 260-epoch run). LR schedule is handed to the scheduler only after `epoch >= warmup_epochs`.

**Gradient clipping:** `max_grad_norm=0.5` (with `ai_system`), `1.0` (MoE only). `clip_grad_norm_` applied before each optimizer step to all params with `.grad is not None`.

**Mixed precision:** `torch.cuda.amp.GradScaler()` + `autocast`. `scaler.unscale_()` before gradient clipping. `scaler.step()` + `scaler.update()` per step.

**Gradient accumulation:** configurable `grad_accum_steps` (2/3/4 based on VRAM). Optimizer step only when `(batch_idx + 1) % grad_accum_steps == 0`.

**Loss NaN/Inf guard (`safe_loss`):** Clamps loss to `max_val=50.0` before backward. If NaN/Inf after clamping, batch is skipped with no gradient update.

**`AdaptiveTrainingSystem`** — Monitors loss variance and convergence rate over a 10-step window:
- `convergence_rate < 0.01` → LR × 0.9
- `convergence_rate > 0.05` → LR × 1.05 (capped at 1e-3)
- `stability_score < 0.5` → LR × 0.85

**`AutomaticHyperparameterOptimizer`** — Bayesian-style search over `lr` (log scale, 1e-5–1e-3), `batch_size` (2/4/8/16), `weight_decay` (log, 1e-6–1e-4), `beta1` (0.85–0.95), `beta2` (0.99–0.9999), `dropout` (0–0.3). Explore/exploit: random search for first 5 trials, then weighted average of top-5 performers with variance-adaptive perturbation (±15% high-variance, ±5% low-variance).

**`AdvancedContinuousLearning`** — Updates expert weights based on observed performance without full retraining.

**Checkpointing:**
- Every 5 epochs (raised from 10 for OOM protection)
- Milestones: 40, 80, 120, 160, 200, 230, 260
- Keep last 30 checkpoints
- Emergency checkpoint on every OOM or unrecoverable error
- `--auto_resume` flag reads `find_latest_checkpoint()` (searches `epoch_*.pt`, `emergency_epoch_*.pt`, `emergency_oom_epoch_*.pt`, `best_model*.pt`, `checkpoint_*.pt` patterns)

```bash
# Basic
python ai_system_core.py --train --epochs 100 --batch_size 6

# Full configuration
python ai_system_core.py \
  --train \
  --data_dir ./data/train \
  --val_dir  ./data/val \
  --epochs   260 \
  --batch_size 2 \
  --save_path ./models/best_model.pt
```

---

## Inference

```bash
python ai_system_core.py \
  --generate \
  --model   ./models/best_model.pt \
  --prompt  "manga girl, forest background" \
  --output  result.png
```

Quality enhance modes: `full` / `resolution` / `detail` / `color`. Staged enhancement (`full_quality_enhance=True`) downscales to VRAM-safe `max_side` before QualityEnhancer, upscales back. Blend factor (`full_enhance_blend`) mixes enhanced and original.

---

## Monitoring and Logging

```python
visualize_training_progress(ai_system, save_path='progress.png')  # 3×3 metric chart
print_progress_dashboard(ai_system)                                # Live console dashboard
```

**TensorBoard:** `tensorboard --logdir=runs/multiverse_origin`

**Tracked metrics:** total loss, per-expert losses (line art / color / background), art theory losses (anatomy / perspective / color harmony), self-critique score, creativity score, PSNR, SSIM, accuracy.

**Validation metrics** (`ValidationMetrics`): rolling `deque(maxlen=1000)` for PSNR, SSIM, accuracy. Running average and latest value accessible per metric.

| Log | Content |
|-----|---------|
| `main.log` | Training events and system status |
| `performance.log` | Per-batch timing and resource utilization |
| `error.log` | Exceptions with full context and recovery status |
| `diagnosis.log` | Per-component health check results |

---

## Project Structure

```
./
├── ai_system_core.py           # Complete system — single file, ~10,000 lines
├── data/
│   └── train/
│       ├── image001.png
│       ├── image001.txt        # Optional caption — paired by filename stem
│       └── ...
├── models/
│   └── best_model.pt
├── cache/
│   └── preprocessed/           # Auto-generated NVMe tensor cache (.pt files)
├── checkpoints/                # Per-epoch and emergency checkpoints
│   ├── epoch_0040.pt
│   ├── emergency_oom_epoch_0085.pt
│   └── ...
├── runs/
│   └── multiverse_origin/      # TensorBoard logs
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

## Intellectual Property

**© 2025–2026 Multiverse Origin Studio. All rights reserved.**

Developed independently by a solo developer. No pretrained image generation backbone.

The following are claimed as original work:

- The complete end-to-end architecture as implemented: 14-expert MoE with two-pass cross-expert attention (16 heads), GNN v4 with 8-head attention per layer connecting expert outputs as graph nodes, `DataFlow` runtime computation graph wiring, full post-processing chain, and all integration as a single-file training and inference system
- `EmotionalArtisticUnderstanding` — direct emotion score (10-category) to generation embedding bridge via concatenation and learned projection, combined with five artistic element extractors and full fusion
- `DeepUnderstandingSystem` — three-level (semantic + 6-layer TransformerEncoder + 4-level hierarchical) understanding stack with relationship network and multi-head attention fusion
- `AdvancedCompressionSystem` — importance-weighted embedding compression with 8-bit inference quantization
- `EnhancedResourceManager` with `L3CacheManager` — hardware-specific allocation budgets, staged quality enhancement (VRAM-neutral for large outputs), and GNN output caching at inference time
- Training methodology as implemented: GAN with label smoothing on an anime-specific discriminator, AMSGrad on both generator and discriminator, temporal consistency loss with outpainting overlap zones and attention-weighted edge loss for panel generation, composite loss stack with NaN/inf guard, OOM-recovery batch-size reduction with automatic checkpoint and resume
- `ArtDataset` with `_create_image_stat_embedding` as a device-safe DataLoader-compatible image representation, `_create_hash_embedding` as a reproducible non-random text fallback, and lazy NVMe tensor caching

This repository is private. Access does not grant any license to use, reproduce, modify, or build upon any part of this system.

---

*Multiverse Origin Studio — All rights reserved*
