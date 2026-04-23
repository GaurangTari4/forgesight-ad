# ForgeSight AD: Industrial Anomaly Detection Studio

ForgeSight AD is an industrial anomaly detection studio that lets users choose the best ResNet backbone (`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`) based on their own hardware and requirements.

It is designed for unsupervised anomaly detection on MVTec AD, using the existing feature-statistics pipeline in your workspace.

## Why this repo exists

Different systems have different constraints:
- Low VRAM laptops need smaller models
- Workstations with large GPUs can run deeper models
- Some teams need speed, others need higher anomaly discrimination

This repo makes that choice explicit, reproducible, and documented.

## Repository Contents

- `choose_backbone.py`: advisory backbone recommendation based on VRAM + latency/priority goals
- `run_pipeline.py`: runs either a single selected backbone or multi-backbone comparison
- `orchestrated_pipeline.py`: one-command workflow runner (`quick_check`, `full_compare`, `demo`)
- `demo_app.py`: Gradio upload demo for heatmap + anomaly highlight visualization
- `configs/system_profiles.json`: example system presets
- `requirements.txt`: Python dependencies

## What changes when the backbone changes

Switching backbone changes these practical aspects:

1. Memory usage (VRAM/RAM)
- `resnet18` < `resnet34` < `resnet50` < `resnet101` < `resnet152`
- Deeper models consume more memory for activations and feature maps.

2. Training and inference time
- Smaller backbones are faster.
- Larger backbones are slower but can improve feature richness.

3. Throughput (images/sec)
- `resnet18` and `resnet34` provide best throughput.
- `resnet101`/`resnet152` trade throughput for representational depth.

4. Accuracy tendencies (not guaranteed)
- Mid/large backbones often improve image-level ROC-AUC on complex categories.
- Gains can plateau; bigger is not always better for every category.

5. Deployment cost
- Larger backbones increase compute cost and latency.
- Smaller backbones are easier for edge or near-real-time deployment.

## Model Selection Guide (Advisory, Not Strict)

These are recommended defaults, not hard limits. You can often run a larger backbone on lower VRAM by reducing `--batch-size` and/or `--image-size`.

- Choose `resnet18` if:
  - VRAM <= 6 GB
  - You prioritize speed/throughput
  - You need lightweight deployment

- Choose `resnet34` if:
  - VRAM is modest (6-8 GB)
  - You want a speed/accuracy middle ground

- Choose `resnet50` if:
  - VRAM >= 8 GB
  - You want a strong default balance

- Choose `resnet101` if:
  - VRAM >= 10 GB
  - You optimize for quality over speed

- Choose `resnet152` if:
  - VRAM >= 12 GB
  - You can afford slower runs and want maximum depth

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Get recommendation from your system/requirements:

```bash
python choose_backbone.py --vram-gb 8 --priority balanced --latency-target-ms 200
```

3. Run selected backbone:

```bash
python run_pipeline.py --mode single --backbone resnet50 --image-size 256 --batch-size 8
```

4. Run full comparison:

```bash
python run_pipeline.py --mode compare --backbones resnet18,resnet34,resnet50,resnet101,resnet152
```

5. Launch upload demo (after models are trained):

```bash
python demo_app.py
```

## One-Command Orchestrated Pipeline

If you do not want to run multiple terminal commands, use:

```bash
python orchestrated_pipeline.py --workflow quick_check --vram-gb 8 --priority balanced
```

Other workflows:

```bash
# Full backbone comparison (all supported ResNets)
python orchestrated_pipeline.py --workflow full_compare --image-size 256 --batch-size 8

# Launch demo app directly
python orchestrated_pipeline.py --workflow demo
```

## Inputs users can control

- Hardware proxy:
  - `--vram-gb`
- Product requirement:
  - `--priority` (`speed`, `balanced`, `accuracy`)
  - `--latency-target-ms`
- Pipeline controls:
  - `--image-size`
  - `--batch-size`
  - `--categories`

## Use Your Own Dataset

You can use any custom industrial dataset as long as you follow the MVTec-style folder layout below.

### Required Folder Structure

```text
your_dataset_root/
  category_a/
    train/
      good/
        000.png
        001.png
    test/
      good/
        100.png
      defect_type_1/
        200.png
      defect_type_2/
        300.png
    ground_truth/
      defect_type_1/
        200_mask.png
      defect_type_2/
        300_mask.png
```

### Important Rules

1. Train set must contain only normal images in `train/good`.
2. Test set should contain both `good` and defect folders.
3. For each defect test image `abc.png`, the mask file must be named `abc_mask.png`.
4. Mask images should be single-channel binary masks (0 background, 255 defect region).
5. Images are expected as `.png` files.

### Run on Custom Dataset

```bash
# Single backbone run
python run_pipeline.py --mode single --backbone resnet101 --dataset-root "<ABS_PATH_TO_DATASET_ROOT>" --categories category_a

# Multi-backbone comparison
python run_pipeline.py --mode compare --dataset-root "<ABS_PATH_TO_DATASET_ROOT>"

# One-command orchestrated quick check
python orchestrated_pipeline.py --workflow quick_check --dataset-root "<ABS_PATH_TO_DATASET_ROOT>" --priority balanced
```

### Demo with Custom Dataset Artifacts

After training completes, `demo_app.py` will automatically list backbones/categories from generated artifacts.

## Output artifacts

The run writes artifacts to your existing comparison pipeline output folder (or custom path):
- per-backbone per-category metrics
- overall leaderboard
- markdown summary report
- demo-ready `model.pt` artifacts per backbone/category


## Troubleshooting

- `AttributeError: Can't pickle local object ... lambda` on Windows:
  - Use the latest code in this repo (transform lambdas were replaced by picklable callables).
- CUDA OOM for larger backbones:
  - Lower `--batch-size` first, then lower `--image-size`.
- No categories found:
  - Confirm `--dataset-root` points to the directory containing category folders.

