# Powerline Sleeve Detection System

A computer vision system for detecting powerline sleeves from street-level imagery.

## Installation

```bash
# Clone the repository
git clone https://github.com/DanDRob/powerline-sleeve-detection.git
cd powerline-sleeve-detection

# Install the package
pip install -e .
```

## Configuration

Before running the system, you need to set up your configuration:

1. Copy the `config.yaml` file and modify it with your settings
2. Set your Google API key in the configuration file or as an environment variable:

```bash
export POWERLINE_API_KEY=your_google_api_key
```

## Usage

### Command Line Interface

Process a single route:

```bash
python run.py route --id "route1" --start "Toronto, ON" --end "Mississauga, ON"
```

Process multiple routes from a CSV file:

```bash
python run.py batch --csv routes.csv --parallel --max-concurrent 3
```

### CSV Format for Batch Processing

Create a CSV file with the following format:

```
route_id,start_location,end_location
route1,Toronto ON,Mississauga ON
route2,Hamilton ON,Burlington ON
```

## Project Structure

- `powerline_sleeve_detection/`: Main package
  - `acquisition/`: Handles route planning and image acquisition
  - `detection/`: Object detection models and logic
  - `processing/`: Batch processing and workflow coordination
  - `system/`: Core system components (config, logging)
  - `visualization/`: Map and dashboard generation

## Output

Results are saved to the configured output directory (default: `output/`):

- Detection images: `output/detections/`
- Maps and visualizations: `output/maps/`
- Reports: `output/reports/`
- Logs: `output/logs/`

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest tests/
```
