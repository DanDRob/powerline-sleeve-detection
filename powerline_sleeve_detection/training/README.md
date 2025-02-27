# Powerline Sleeve Detection Training Module

This module provides tools for training and fine-tuning custom YOLOv8 models for powerline sleeve detection.

## Features

- **Dataset preparation**: Tools for organizing and preparing data for training
- **Data augmentation**: Enhance datasets with synthetic variations to improve model robustness
- **Model training**: Fine-tune YOLOv8 models for sleeve detection
- **Hyperparameter tuning**: Automated optimization of model parameters
- **Automated labeling**: Tools for automatically labeling images with trained models
- **Model ensembling**: Combine multiple models for improved detection performance
- **Training monitoring**: Visualize and track training progress

## Workflow

The recommended workflow for training a custom sleeve detection model is:

1. **Collect and label data**

   - Place raw images in `dataset/raw/images/`
   - Create YOLO format labels in `dataset/raw/labels/`
   - Alternatively, use auto-labeling for initial labels

2. **Prepare the dataset**

   - Split data into train/val/test sets
   - Organize according to YOLO format

3. **Augment the dataset** (optional)

   - Create synthetic variations to increase dataset size and diversity

4. **Train the model**

   - Fine-tune a pre-trained YOLOv8 model
   - Monitor training progress and validation metrics

5. **Optimize hyperparameters** (optional)

   - Run automated hyperparameter search
   - Train with optimal parameters

6. **Create ensembles** (optional)
   - Combine multiple trained models for improved performance

## Command Line Interface

All functionality is accessible through the command line interface:

### Dataset Preparation

```bash
# Create an empty dataset structure
python -m powerline_sleeve_detection create-empty-dataset --class-names sleeve

# Prepare a dataset from raw data
python -m powerline_sleeve_detection prepare-dataset --data dataset/raw --splits 0.7 0.15 0.15 --single-class

# Augment a dataset
python -m powerline_sleeve_detection augment-dataset --dataset dataset/train --multiplier 3 --severity medium
```

### Training

```bash
# Train a model
python -m powerline_sleeve_detection train --base-model yolov8n.pt --dataset dataset.yaml --epochs 100 --batch-size 16

# Perform hyperparameter tuning
python -m powerline_sleeve_detection tune --base-model yolov8n.pt --dataset dataset.yaml --trials 20
```

### Auto-labeling

```bash
# Automatically label images using a trained model
python -m powerline_sleeve_detection autolabel --model models/transmission_model.pt --images unlabeled_data --output labeled_data

# Semi-supervised labeling with human review
python -m powerline_sleeve_detection semilabel --model models/transmission_model.pt --images unlabeled_data --output labeled_data
```

### Ensemble Creation and Evaluation

```bash
# Create an ensemble of models
python -m powerline_sleeve_detection create-ensemble --models model1.pt,model2.pt,model3.pt --weights 1.0,0.8,0.6 --output-config ensemble.yaml

# Evaluate ensemble performance
python -m powerline_sleeve_detection evaluate-ensemble --test-dir test_images --ground-truth-dir test_labels
```

## Data Format

### YOLO Format Labels

Labels should follow the YOLO format:

```
<class_id> <x_center> <y_center> <width> <height>
```

Where:

- `class_id`: Integer class identifier (0 for sleeve in single-class mode)
- `x_center`, `y_center`: Normalized center coordinates (0.0-1.0)
- `width`, `height`: Normalized width and height (0.0-1.0)

Each `.txt` label file should have the same name as its corresponding image file.

## Configuration

Training parameters can be configured in `config.yaml`:

```yaml
training:
  dataset_dir: "data/sleeves"
  batch_size: 16
  epochs: 100
  patience: 20
  learning_rate: 0.01
  base_model: "yolov8s.pt"
  custom_model_path: "models/transmission_model.pt"
  # See config.yaml for all available options
```

## Tips for Better Models

1. **Diverse training data**: Include images with different lighting conditions, backgrounds, and sleeve orientations
2. **Balanced classes**: Ensure balanced representation of different sleeve types if using multiple classes
3. **Quality over quantity**: Having fewer high-quality labels is better than many poor labels
4. **Careful validation**: Use a representative validation set to monitor for overfitting
5. **Ensemble approach**: Combine multiple models trained with different configurations for best results
