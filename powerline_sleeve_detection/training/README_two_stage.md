# Two-Stage Detection for Powerline Sleeves

This module implements a two-stage object detection approach for finding sleeves on powerlines:

1. First, detect powerlines in images
2. Then, detect sleeves within the powerline regions

This approach is more effective than directly detecting sleeves because:

- Powerlines are easier to detect than sleeves (larger, more consistent appearance)
- Narrowing the search area to powerlines significantly reduces false positives for sleeves
- Image regions containing powerlines can be processed at higher resolution

## Workflow

The complete workflow consists of these steps:

### Stage 1: Powerline Detection

1. Setup the directory structure
2. Label powerlines in your images
3. Augment the labeled dataset
4. Train a powerline detection model
5. Detect powerlines in your images

### Stage 2: Sleeve Detection

6. Extract powerline regions from detection results
7. Label sleeves in the extracted regions
8. Augment the labeled sleeve dataset
9. Train a sleeve detection model
10. Run the complete pipeline

## Usage Examples

### 1. Setup Directory Structure

```bash
python run.py two-stage --mode setup --base-dir data/two_stage
```

This creates the necessary directory structure:

```
data/two_stage/
├── powerline/
│   ├── raw/            # Raw images
│   ├── labeled/        # Manually labeled powerlines
│   ├── augmented/      # Augmented powerline dataset
│   ├── datasets/       # Training/validation splits
│   ├── models/         # Trained powerline models
│   └── results/        # Detection results
└── sleeve/
    ├── raw/            # Extracted powerline regions
    ├── labeled/        # Manually labeled sleeves
    ├── augmented/      # Augmented sleeve dataset
    ├── datasets/       # Training/validation splits
    ├── models/         # Trained sleeve models
    └── results/        # Detection results
```

### 2. Label Powerlines

```bash
python run.py two-stage --mode label-powerlines --images path/to/raw/images --output data/two_stage/powerline/labeled
```

This sets up the environment for manual labeling with LabelImg and provides instructions.

### 3. Augment Powerline Dataset

```bash
python run.py two-stage --mode augment-powerlines --images data/two_stage/powerline/labeled --output data/two_stage/powerline/augmented --augmentations 5 --severity medium
```

This creates 5 augmented versions of each labeled image, with moderate transformations.

### 4. Train Powerline Detection Model

```bash
python run.py two-stage --mode train-powerlines --images data/two_stage/powerline/augmented --output data/two_stage/powerline/models --epochs 100 --batch-size 16 --model-size m
```

This trains a YOLOv5m model on the augmented powerline dataset.

### 5. Detect Powerlines in Images

```bash
python run.py two-stage --mode detect-powerlines --images path/to/raw/images --output data/two_stage/powerline/results --powerline-model data/two_stage/powerline/models/powerline_detector_m/weights/best.pt --conf-threshold 0.25
```

This runs inference using the trained powerline model.

### 6. Extract Powerline Regions

```bash
python run.py two-stage --mode extract-regions --images data/two_stage/powerline/results --output data/two_stage/sleeve/raw --padding 0.1
```

This extracts regions around detected powerlines with 10% padding.

### 7. Label Sleeves

```bash
python run.py two-stage --mode label-sleeves --images data/two_stage/sleeve/raw --output data/two_stage/sleeve/labeled
```

This sets up the environment for manual labeling of sleeves.

### 8. Augment Sleeve Dataset

```bash
python run.py two-stage --mode augment-sleeves --images data/two_stage/sleeve/labeled --output data/two_stage/sleeve/augmented --augmentations 5 --severity medium
```

This creates 5 augmented versions of each labeled sleeve image.

### 9. Train Sleeve Detection Model

```bash
python run.py two-stage --mode train-sleeves --images data/two_stage/sleeve/augmented --output data/two_stage/sleeve/models --epochs 100 --batch-size 16 --model-size m
```

This trains a YOLOv5m model on the augmented sleeve dataset.

### 10. Run Complete Pipeline

```bash
python run.py two-stage --mode run-pipeline --images path/to/raw/images --output data/two_stage/results --powerline-model data/two_stage/powerline/models/powerline_detector_m/weights/best.pt --sleeve-model data/two_stage/sleeve/models/sleeve_detector_m/weights/best.pt
```

This runs the complete two-stage detection pipeline on new images.

## Manual Labeling Instructions

For manual labeling, we use LabelImg, a graphical image annotation tool:

1. Install LabelImg: `pip install labelImg`
2. Run LabelImg with the command provided after setup
3. Use the following keyboard shortcuts for faster labeling:
   - W: Create a rectangle box
   - D: Next image
   - A: Previous image
   - Ctrl+S: Save
   - Ctrl+R: Change default save directory

When labeling powerlines:

- Draw bounding boxes around the entire visible powerline segment
- Use a single class: "powerline"

When labeling sleeves:

- Draw tight bounding boxes around each sleeve
- Use a single class: "sleeve"

## Model Selection

For optimal results, we use YOLOv5 models of different sizes:

- YOLOv5n: Fastest, smallest, less accurate
- YOLOv5s: Good balance for smaller/simpler objects
- YOLOv5m: Recommended default for powerline detection
- YOLOv5l: Better accuracy, slower inference
- YOLOv5x: Highest accuracy, slowest inference

## Image Augmentation

The augmentation pipeline generates additional training examples with these transformations:

- Geometric: flips, rotations, scaling, perspective changes
- Color/intensity: brightness, contrast, hue, saturation adjustments
- Noise/artifacts: blur, shadows, compression artifacts
- Weather effects (at heavy severity): snow, rain, fog

The number of augmentations and severity can be adjusted to suit your dataset.

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Albumentations
- YOLOv5 (automatically cloned during training)
- LabelImg (for manual annotation)
