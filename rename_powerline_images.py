import os
import shutil
from pathlib import Path
import re

# Source and destination directories
src_base_dir = Path("output/batch_results")
dst_dir = Path("data/two_stage/powerline/raw")

# Create destination directory if it doesn't exist
os.makedirs(dst_dir, exist_ok=True)

# Regular expression to extract information from filenames
pattern = r"snapshot_(?P<direction>left|right)_(?P<point>\d+)_heading_(?P<heading>\d+)_pitch_(?P<pitch>\d+)_fov_(?P<fov>\d+)\.jpg"

# Counter for files processed
total_files = 0
processed_files = 0
route_counters = {}

# Process each route directory
for route_dir in src_base_dir.glob("route_*"):
    route_num = route_dir.name.split("_")[1]
    images_dir = route_dir / "images"

    # Initialize route counter
    route_counters[route_num] = 0

    if not images_dir.exists():
        print(f"No images directory found in {route_dir}")
        continue

    print(f"Processing route {route_num}...")

    # Process each image file
    for src_file in images_dir.glob("*.jpg"):
        total_files += 1
        filename = src_file.name

        # Extract information using regex
        match = re.match(pattern, filename)
        if match:
            # Extract components
            direction = match.group("direction")
            point = match.group("point")
            heading = match.group("heading")
            pitch = match.group("pitch")
            fov = match.group("fov")

            # Create new filename
            new_filename = f"pl_r{route_num}_p{point}_{direction}_h{heading}_p{pitch}_fov{fov}.jpg"
            dst_file = dst_dir / new_filename

            # Copy file with new name
            shutil.copy2(src_file, dst_file)
            processed_files += 1
            route_counters[route_num] += 1
            print(f"Copied: {filename} -> {new_filename}")
        else:
            print(f"Skipping file with unmatched pattern: {filename}")

print(f"Processed {processed_files} out of {total_files} files")
for route, count in route_counters.items():
    print(f"Route {route}: {count} files processed")
print(f"All files copied to {dst_dir}")
