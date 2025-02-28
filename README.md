# Powerline Sleeve Detection System

A robust system for detecting sleeves on powerlines from aerial imagery and video footage. This project combines computer vision, deep learning (YOLOv8), and data processing capabilities to identify powerline sleeves with high accuracy.

## Features

- **Detection Pipeline**: Process video footage to detect and track power line sleeves
- **Ensemble Model Support**: Combine multiple models for improved detection accuracy
- **Route Planning**: Plan optimal routes for data acquisition
- **Batch Processing**: Process multiple routes in parallel
- **Training Framework**: Complete training infrastructure for custom sleeve detection models
- **Model Evaluation**: Comprehensive tools for evaluating model performance
- **Data Augmentation**: Enhance training data with various transformations
- **Hyperparameter Optimization**: Find optimal parameters for training models
- **Auto-Labeling**: Assist with dataset creation through automatic labeling

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/powerline-sleeve-detection.git
cd powerline-sleeve-detection

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Usage

The system provides a command-line interface for various operations:

### Processing Routes

```bash
# Process a single route
powerline-detector process --route-id ROUTE_ID

# Process multiple routes from a CSV file
powerline-detector process --batch routes.csv --parallel --max-tasks 4

# Process a list of videos
powerline-detector process --video-list videos.csv
```

### Planning Routes

```bash
# Plan a single route
powerline-detector plan --route-id ROUTE_ID

# Plan multiple routes from a CSV file
powerline-detector plan --batch routes.csv
```

### Training Models

```bash
# Prepare a dataset
powerline-detector train --mode prepare --data /path/to/raw/data

# Augment a dataset
powerline-detector train --mode augment --data /path/to/dataset --multiplier 3

# Train a model
powerline-detector train --mode train --dataset-yaml /path/to/dataset.yaml --base-model yolov8n.pt --epochs 100

# Tune hyperparameters
powerline-detector train --mode tune --dataset-yaml /path/to/dataset.yaml --trials 20

# Create an ensemble
powerline-detector train --mode create-ensemble --model-paths /path/to/model1.pt /path/to/model2.pt

# Run a complete workflow
powerline-detector train --mode full-workflow --data /path/to/raw/data --base-model yolov8n.pt
```

### Evaluating Models

```bash
# Evaluate a single model
powerline-detector train --mode evaluate --eval-mode single --model-path /path/to/model.pt --test-dir /path/to/test/images

# Evaluate an ensemble
powerline-detector train --mode evaluate --eval-mode ensemble --model-paths /path/to/model1.pt /path/to/model2.pt --test-dir /path/to/test/images

# Compare multiple models
powerline-detector train --mode evaluate --eval-mode compare --model-paths /path/to/model1.pt /path/to/model2.pt --model-names "Model A" "Model B" --test-dir /path/to/test/images
```

## Configuration

The system is configured using a YAML file. The default configuration file is `config.yaml` in the current directory. You can specify a different configuration file using the `--config` option.

Key configuration sections include:

- `system`: General system settings
- `detector`: Detection parameters
- `tracking`: Tracking parameters
- `training`: Training parameters
- `augmentation`: Data augmentation settings
- `hyperparameter_tuning`: Hyperparameter optimization settings
- `auto_labeling`: Automatic labeling settings
- `ensemble`: Ensemble model settings

## Data Format

### Route CSV Format

For batch processing, the CSV file should have the following columns:

- `route_id`: Unique identifier for the route
- `start`: Start location (address or lat,lng)
- `end`: End location (address or lat,lng)

### Video List CSV Format

For video processing, the CSV file should have the following columns:

- `video_path`: Path to the video file
- `route_id`: Unique identifier for the route (optional)
- `timestamp`: Timestamp of the video (optional)

### Dataset Format

The dataset should follow the YOLOv8 format:

- `images/train/`: Training images
- `images/val/`: Validation images
- `images/test/`: Test images
- `labels/train/`: Training labels
- `labels/val/`: Validation labels
- `labels/test/`: Test labels
- `dataset.yaml`: Dataset configuration file

## Project Structure

- `powerline_sleeve_detection/`: Main package
  - `acquisition/`: Route planning and data acquisition
  - `processing/`: Video processing and detection
  - `training/`: Model training and evaluation
  - `system/`: Common utilities and configuration
  - `cli.py`: Command-line interface

## Development

### Running Tests

```bash
pytest
```

### Building Documentation

```bash
cd docs
make html
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- YOLOv8 by Ultralytics
- OpenCV for computer vision capabilities
- PyTorch for deep learning
