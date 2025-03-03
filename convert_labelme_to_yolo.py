#!/usr/bin/env python3
import os
import json
import glob
from pathlib import Path
import argparse


def convert_labelme_to_yolo(json_dir, output_dir, class_mapping=None):
    """
    Convert LabelMe JSON annotations to YOLO format.

    Args:
        json_dir: Directory containing LabelMe JSON files
        output_dir: Output directory for YOLO txt files
        class_mapping: Dictionary mapping class names to indices (defaults to {'powerline': 0})
    """
    if class_mapping is None:
        class_mapping = {'powerline': 0}

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files to convert")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                json_data = json.load(f)

            image_width = json_data['imageWidth']
            image_height = json_data['imageHeight']

            # Get base filename without extension
            base_name = os.path.basename(json_file).replace('.json', '')

            yolo_lines = []

            # Process each shape
            for shape in json_data['shapes']:
                label = shape['label']
                if label not in class_mapping:
                    print(
                        f"Warning: Label '{label}' not in class mapping, skipping")
                    continue

                class_id = class_mapping[label]

                # Get shape points
                if shape['shape_type'] == 'polygon' or shape['shape_type'] == 'rectangle':
                    points = shape['points']

                    # Convert rectangle to YOLO format (x_center, y_center, width, height)
                    if shape['shape_type'] == 'rectangle' or len(points) == 2:
                        # Rectangle is stored as [top-left, bottom-right]
                        if len(points) == 2:  # Standard rectangle format
                            x1, y1 = points[0]
                            x2, y2 = points[1]
                        else:  # Polygon with 4 points (less common)
                            x_coords = [p[0] for p in points]
                            y_coords = [p[1] for p in points]
                            x1, y1 = min(x_coords), min(y_coords)
                            x2, y2 = max(x_coords), max(y_coords)

                        # Calculate YOLO format
                        x_center = (x1 + x2) / 2.0
                        y_center = (y1 + y2) / 2.0
                        width = x2 - x1
                        height = y2 - y1

                        # Normalize to 0-1
                        x_center /= image_width
                        y_center /= image_height
                        width /= image_width
                        height /= image_height

                        # Add to YOLO lines
                        yolo_lines.append(
                            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                    # For polygon shapes, convert to bounding box
                    elif shape['shape_type'] == 'polygon':
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]

                        # Get bounding box
                        x1, y1 = min(x_coords), min(y_coords)
                        x2, y2 = max(x_coords), max(y_coords)

                        # Calculate YOLO format
                        x_center = (x1 + x2) / 2.0
                        y_center = (y1 + y2) / 2.0
                        width = x2 - x1
                        height = y2 - y1

                        # Normalize to 0-1
                        x_center /= image_width
                        y_center /= image_height
                        width /= image_width
                        height /= image_height

                        # Add to YOLO lines
                        yolo_lines.append(
                            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Write YOLO file
            output_file = os.path.join(output_dir, f"{base_name}.txt")
            with open(output_file, 'w') as f:
                f.write("\n".join(yolo_lines))

            print(f"Converted {json_file} -> {output_file}")

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    print("Conversion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LabelMe JSON annotations to YOLO format")
    parser.add_argument("--json_dir", required=True,
                        help="Directory containing LabelMe JSON files")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for YOLO txt files")
    args = parser.parse_args()

    convert_labelme_to_yolo(args.json_dir, args.output_dir)
