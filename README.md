# Multiverse Origin Studio AI

> **All rights reserved — Multiverse Origin Studio**  
> Built from scratch by a solo developer. No pretrained image generation backbone.  
> Single-file architecture. Verified training on mid-range consumer hardware.

---

**Multiverse Origin Studio AI is a complete end-to-end anime and manga image generation system built entirely from scratch — architecture, loss functions, data pipeline, and resource management — designed to produce high-quality output on hardware that existing systems treat as insufficient.**

This is not a wrapper, fine-tune, or derivative of Stable Diffusion, Midjourney, or any other existing image model. Every component was designed knowing what every other component requires.

**Verified result:** Successfully trained to output on 2,000+ curated images using 16GB RAM + RTX 3060 12GB VRAM. Estimated training time: 3–4 days continuous. Domain-specific losses with quality-filtered data outperform brute-force approaches requiring 10,000–50,000+ images.

---

## Tested Hardware

| Component | Specification |
|-----------|--------------|
| CPU | AMD Ryzen 5 5500 — 6 cores / 10 threads, 16MB L3 cache |
| GPU | NVIDIA RTX 3060 — 12GB VRAM |
| RAM | 16 GB |
| Storage | M.2 NVMe SSD — used as persistent tensor cache layer |

The resource management system was built specifically for this hardware profile. Every allocation, cache budget, and batch scaling decision references these numbers directly.

---

## Architecture

### System Controller

**`MultiverseOriginStudioAI`** — Top-level orchestrator. Initializes and wires every subsystem, manages the training loop, handles checkpointing, and coordinates device placement across CPU and GPU. Modules are selectively placed on GPU (quality-critical paths: `EmotionalArtisticUnderstanding`, `DeepUnderstandingSystem`, GNN v1–v4, MoE, diffusion, encoders) or CPU (VRAM-constrained paths: `AdvancedCompressionSystem`, `DesignStyleSupportSystem`, `QualityEnhancer`) to maintain stable VRAM usage throughout multi-day training runs.

A projection layer (`nn.Linear(1024 → 512) → LayerNorm → ReLU`) bridges MoE output dimension to internal module input expectations, maintaining shape compatibility across the pipeline without restructuring the model.

---

### Mixture of Experts — 14 Specialists

**`EnhancedMoE`** routes each generation task through 14 specialized expert agents rather than a single monolithic model. The router is a learned network (`Linear(768→1024) → LayerNorm → GELU → Dropout → Linear(1024→512) → LayerNorm → GELU → Linear(512→14)`) with a learnable temperature parameter that controls exploration vs. exploitation across experts.

Each expert is an `EnhancedExpertAgent` receiving 768-dim input and producing 512-dim output through a hidden dimension of 512.

**Expert responsibilities:**

| Expert | Function |
|--------|---------|
| `line_art` | Edge structure, stroke continuity, Laplacian-based thickness matching |
| `color` | Palette generation, color harmony, distribution and shading consistency |
| `light` | Lighting direction, intensity, shadow and specular surface modeling |
| `anatomy` | Body proportion constraints, bilateral symmetry enforcement |
| `scene` | Spatial composition, layout reasoning |
| `camera` | Camera angle, projection matrix transformation, perspective |
| `story` | Multi-language narrative conditioning (10 languages) |
| `structure` | High-level compositional structure and form |
| `precision` | Detail refinement, local sharpness enhancement |
| `memory` | Temporal consistency across panels and generated sequences |
| `safety` | Output constraint enforcement |
| `editing` | Region-based modification and inpainting support |
| `quality` | Output quality scoring and per-sample enhancement |
| `style` | Style fingerprinting, artist-aware generation, style transfer |

**Two-pass execution:**  
Experts run a first pass independently, producing initial embeddings. These are stacked into a tensor and passed through 16-head cross-expert attention (`nn.MultiheadAttention(expert_dim, num_heads=16)`), allowing each expert to observe what every other found before committing. A coordinator layer applies a residual refinement. Routing weights scale each expert's contribution; a separate fusion head (`Linear(expert_dim × 14 → expert_dim × 4 → expert_dim × 2 → expert_dim)`) processes the concatenated outputs. Final output combines weighted sum and fused projection.

Routing history is tracked in a `deque(maxlen=1000)` for analysis and debugging of expert utilization.

---

### Graph Neural Network — Four Versions

**`EnhancedGNN`** (versions 1–4, 6 layers, multi-head attention) connects expert outputs as graph nodes. Rather than treating expert outputs as independent signals, the GNN enables relational reasoning — the output of one expert can attend to and modify the representation of another through learned message passing.

All four versions (v1–v4) are initialized and available. Version 4 is registered as the active graph module in the `DataFlow` pipeline. Earlier versions are retained for ablation, comparison, and fallback.

---

### Computation Graph — DataFlow

**`DataFlow`** is a thread-safe pipeline registry. Modules register their names, input dependencies, and output names. `DataFlow` resolves the dependency order and executes modules in the correct sequence. The computation graph can be rewired at runtime without restructuring the model or rewriting forward passes.

Module registration example:
```python
data_flow.register_module('line_art', line_art_expert,
    inputs=['embedding', 'structure'],
    outputs=['line_art'])
```

This architecture allows new experts or processing stages to be inserted into the pipeline without modifying existing module code.

---

### Diffusion Model

**`DiffusionModel`** — U-Net architecture with sinusoidal time embedding and cross-attention text conditioning. Text embeddings (768-dim) condition generation at multiple U-Net scales via cross-attention, allowing the prompt to influence structure and detail simultaneously rather than only at the bottleneck.

---

### Encoders

**`TextEncoder`** — Uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for real semantic embeddings across 10 languages. Falls back to a deterministic SHA-256 hash-based encoding (not random initialization) when the transformer model is unavailable, preserving output reproducibility across runs without the transformer dependency.

**`VisionEncoder`** — Multi-scale CNN feature extraction producing three distinct embedding types:
- Fine texture features via 3×3 convolutions
- Mid-level structural features via pooled intermediate representations  
- Semantic summary via `AdaptiveAvgPool`

All three scales are returned separately, allowing downstream modules to select the appropriate level of abstraction for their task.

---

### Post-Processing Pipeline

After the diffusion model generates an image, output passes through a dedicated quality chain:

**`SuperResolutionSystem`** — 8-block residual SRGAN with PixelShuffle upsampling. Supports 2× and 4× upscaling. Bicubic fallback maintains spatial alignment when the learned upsampler is unavailable.

**`AdvancedDetailEnhancer`** — Three parallel detail extraction branches at 3×3, 5×5, and 7×7 kernel sizes, fused with dedicated edge and texture enhancement paths. Multi-scale detail extraction prevents the model from over-optimizing for a single frequency range.

**`ColorVibrancySystem`** — Separate learnable parameters for saturation, contrast, and brightness, followed by a color harmony refinement pass. Saturation, contrast, and brightness are decoupled so each can be adjusted independently without coupling artifacts.

**`MultiScaleProcessor`** — Processes the generated image simultaneously at 1×, 2×, and 4× downsampled scales, upsamples all three back to original resolution, and fuses them. Preserves both global structural coherence and fine local detail that single-scale processing loses.

**`QualityEnhancer`** — Final scoring and enhancement stage. Placed on CPU to preserve VRAM during generation.

---

## Loss Functions

Every expert is trained with a loss matched to its domain. Domain-specific losses enforce the constraints that matter for that task, rather than applying a generic pixel reconstruction objective to all outputs.

**`AdvancedLossFunctions`** assembles the full loss stack:

**VGG Perceptual Loss (`VGGPerceptualLoss`)** — Computes loss in VGG feature space rather than pixel space. Forces the model to match perceptual features (texture, style, high-level structure) rather than per-pixel intensity, which produces sharper and more visually coherent output.

**SSIM Loss (`SSIMLoss`)** — Structural Similarity Index loss computed via Gaussian-windowed local statistics. SSIM captures luminance, contrast, and structure independently, making it sensitive to the structural distortions that pixel-based losses miss.

```python
# SSIM: 1 − SSIM for minimization
ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) /
           ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
loss = 1 - ssim_map.mean()
```

**Edge Preservation Loss (`EdgePreservationLoss`)** — Sobel gradient magnitude matching for edge alignment + Laplacian detail matching for high-frequency structure. Combined 60/40:

```python
edge_loss   = F.l1_loss(pred_edges, target_edges)    # Sobel magnitude
detail_loss = F.l1_loss(pred_detail, target_detail)  # Laplacian response
total       = 0.6 * edge_loss + 0.4 * detail_loss
```

**Noise Reduction Loss (`NoiseReductionLoss`)** — Three-component loss targeting noise specifically in smooth regions:
1. Gaussian smoothness loss — smooth areas of prediction match smooth areas of target
2. High-frequency noise loss — 8-neighbor Laplacian response in smooth regions is penalized
3. Spatial variance loss — variance in smooth regions is constrained

Smooth region mask is computed as pixels below the 30th percentile of edge response, ensuring the noise penalty applies only where it should — not where high-frequency content is intentional (edges, line art).

```python
smooth_mask = (target_edges < target_edges.quantile(0.3)).float()
noise_loss  = torch.abs(pred_hf * smooth_mask - target_hf * smooth_mask).mean()
total = 0.4 * smoothness_loss + 0.4 * noise_loss + 0.2 * var_loss
```

**Line Art Loss** — L1 loss over edge structure + Laplacian-based thickness consistency:

```python
edge_matching  = F.l1_loss(line_edges, target_edges)
thickness_loss = F.l1_loss(
    F.conv2d(pred_gray, laplacian, padding=1),
    F.conv2d(target_gray, laplacian, padding=1))
line_art_loss  = 0.6 * edge_matching + 0.4 * thickness_loss
```

**Color Loss** — Color distribution matching + standard deviation matching + shading gradient alignment:

```python
color_dist_loss = F.mse_loss(gen_mean, target_mean)
color_std_loss  = F.mse_loss(gen_std, target_std)
shading_loss    = (F.l1_loss(grad_x, tgt_gx) + F.l1_loss(grad_y, tgt_gy)) / 2
color_loss = 0.4 * color_dist_loss + 0.3 * color_std_loss + 0.3 * shading_loss
```

**Anatomical Loss (`AnatomicalDetector`)** — Separate detection and loss weighting for face, hands, and eyes — the regions where anatomical errors are most perceptually salient:

```python
loss = 0.5 * F.mse_loss(pred_face, target_face)
     + 0.3 * F.mse_loss(pred_hands, target_hands)
     + 0.2 * F.mse_loss(pred_eyes, target_eyes)
```

**Art Theory Losses:**
- **Anatomy** — Bilateral symmetry check + Sobel-based edge proportion matching
- **Perspective** — Gradient magnitude consistency for depth cue alignment
- **Color Harmony** — Penalizes over-saturated or clashing color distributions

**Style Consistency Loss (`StyleConsistencyLoss`)** — Gram matrix matching between generated and target style embeddings, enforcing consistent stylistic identity across generated outputs.

**Anime Style Discriminator (`AnimeStyleDiscriminator`)** — GAN discriminator trained specifically for anime/manga aesthetics rather than photorealistic plausibility. Prevents the generator from drifting toward photorealistic features that degrade anime-style quality.

---

## Resource Management

The resource management system was designed from the ground up for the R5 5500 + RTX 3060 hardware profile. Every limit, buffer size, and threshold is derived from measured hardware capacity rather than generic defaults.

**`EnhancedResourceManager`** — Monitors RAM, VRAM, and CPU usage continuously via `psutil` and `torch.cuda`. Computes dynamic batch size based on available resources and current resolution. Distinguishes OS page cache from actual process memory to avoid false headroom estimation. Triggers cleanup at configurable thresholds:

```
RAM:   soft cleanup at 90%, aggressive at 95%
VRAM:  soft cleanup at 92%, aggressive at 95%
```

Aggressive cleanup: double `gc.collect()` + `cuda.empty_cache()` + `cuda.synchronize()`, then re-checks before proceeding.

Allocation budget:
```
max_ram_gb    = 14.4   # 16GB total, 1.6GB reserved for OS
max_vram_gb   = 11.4   # 12GB total, 0.6GB reserved
cpu_threads   = 10     # 2 reserved for OS and background processes
```

**`L3CacheManager`** — Manages the CPU's 16MB L3 cache via LRU eviction. Budget: 15MB (1MB reserved). Stores up to 1,500 tensor entries (~10KB each). Only caches tensors smaller than 50% of the budget to prevent cache thrashing. Tracks hit rate, miss rate, and eviction count for optimization analysis.

```
L3 total:    16MB (R5 5500)
Budget:      15MB (1MB reserved for OS)
Max entries: 1,500 (~10KB each)
Eviction:    LRU (OrderedDict-based)
```

**`SSDOptimizer`** — NVMe as a persistent tensor cache. Preprocessed image tensors are serialized to `.pt` files on first access and served directly from disk on all subsequent epochs. This eliminates redundant image decoding and CPU transform overhead after the first epoch — every image is decoded and augmented exactly once per dataset build, then served as pre-computed tensors for the remainder of training.

**Thread Management** — DataLoader uses 8 persistent worker processes (10 threads − 2 reserved). `multiprocessing_context='spawn'` for Windows compatibility. `pin_memory=True` when CUDA is available for direct host-to-device DMA transfers.

**Resolution-Aware Batch Scaling** — Batch size scales automatically with output resolution:

```
512px   → full batch
768px   → half batch
1024px  → quarter batch
2048px  → further reduced
```

Prevents VRAM overflow without requiring per-run manual configuration.

**OOM Prevention** — `prevent_oom()` checks RAM and VRAM headroom before any high-memory operation. Falls back to minimum batch size of 1 under critical conditions rather than crashing.

---

## Training System

**`AdaptiveTrainingSystem`** — Monitors convergence rate and loss variance in real time. Reduces learning rate when loss plateaus; increases it slightly when convergence is healthy. Prevents both stagnation and overshooting without manual LR schedule design.

**`AutomaticHyperparameterOptimizer`** — Bayesian-style search over learning rate, batch size, weight decay, and dropout. Runs configurable trial epochs before committing to a configuration for the full training run. Removes the manual grid search step.

**`AdvancedContinuousLearning`** — Enables the model to incorporate new data and feedback without full retraining from scratch. Integrates with the MoE system to update expert weights based on observed performance.

**Checkpointing** — Saved every 10 epochs and at milestones: 50, 100, 150, 200, 250, 300. The 20 most recent checkpoints are retained automatically, with older ones pruned to prevent storage saturation.

```bash
# Training
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

---

## Dataset Collection

Multi-source image downloader with strict content and quality filtering built in. Target dataset size: 3,500 images. Parallel downloads via `ThreadPoolExecutor`.

**Sources:**

| Source | Filter |
|--------|--------|
| Danbooru | Safe rating + 40+ NSFW keyword blacklist |
| Safebooru | Additional rating verification |
| Konachan | Safe rating filter |
| Yande.re | Post-2015 only, score ≥ 3 |
| Zerochan | Anime category |
| Anime-pictures | SFW verified |
| Wallhaven | Anime category, SFW purity flag |
| HuggingFace | `huggan/animeface-dataset` |
| Waifu.im / Waifu.pics / Nekos.best | SFW endpoints only |
| Google CSE | Custom Search API, configurable query |

**Quality filters applied to every image:**
- Minimum resolution: 512px on the longer edge (128px configurable minimum)
- Aspect ratio: between 1:4 and 4:1
- Sharpness: Laplacian variance > 20 (threshold accepts soft anime styles while rejecting blurry images)
- Color variance: standard deviation > 10 (rejects near-blank or single-color images)
- Duplicate detection: full-file MD5 hash checked against all previously downloaded images, computed in parallel via `ThreadPoolExecutor`

If the dataset already contains the target number of images, download is skipped entirely. If it exceeds the target, excess files are removed deterministically before new downloads begin.

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

## Monitoring and Logging

```python
visualize_training_progress(ai_system, save_path='progress.png')  # 3×3 metric chart
print_progress_dashboard(ai_system)                                # Live console dashboard
```

**Tracked metrics:** total loss, per-expert losses (line art / color / background), art theory losses (anatomy / perspective / color harmony), self-critique score, creativity score.

**Log files:**

| File | Content |
|------|---------|
| `main.log` | General training events and system status |
| `performance.log` | Per-batch timing and resource utilization |
| `error.log` | Exceptions with full context and recovery status |
| `diagnosis.log` | Automatic health check results per component |

---

## Project Structure

```
./
├── ai_system_core.py          # Complete system — single file, ~10,000 lines
├── data/
│   └── train/
│       ├── image001.jpg
│       ├── image001.txt       # Optional caption (paired by filename stem)
│       └── ...
├── models/
│   └── best_model.pt
├── cache/
│   └── preprocessed/          # Auto-generated NVMe tensor cache
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
