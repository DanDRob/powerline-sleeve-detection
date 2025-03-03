# Guide to Annotating Images with LabelMe

This guide explains how to use LabelMe to annotate your powerline images and convert the annotations to the YOLO format required by the training system.

## 1. Starting LabelMe

Run the following command in the terminal:

```
labelme data\two_stage\powerline\labeled\images --labels data\two_stage\powerline\labeled\classes.txt --nodata
```

This will launch the LabelMe interface with your images and the class labels loaded.

## 2. Annotating Images

### Basic Controls

- **Navigate Between Images**: Use the file list on the left side or the arrow keys
- **Zoom**: Use the mouse wheel to zoom in and out
- **Pan**: Click and hold the middle mouse button to pan the image

### Creating Annotations

1. **Create a Rectangle Around a Powerline**:

   - Click the "Create Rectangle" button in the toolbar (or press the 'R' key)
   - Click and drag to draw a rectangle around the powerline
   - When you release the mouse, you'll be prompted to choose a label
   - Select "powerline" from the dropdown menu
   - Click "OK"

2. **Create a Polygon** (for more precise annotations):

   - Click the "Create Polygon" button in the toolbar (or press the 'P' key)
   - Click points around the outline of the powerline
   - Press Enter or double-click to complete the polygon
   - Select "powerline" from the dropdown menu
   - Click "OK"

3. **Edit an Annotation**:

   - Click on an existing annotation to select it
   - Drag the control points to resize/reshape it
   - Right-click on it for more options (Delete, Edit Label, etc.)

4. **Save Your Work**:
   - LabelMe automatically saves annotations as JSON files in the same directory as the images
   - Each image will have a corresponding .json file

## 3. Converting Annotations to YOLO Format

After you've annotated your images, you need to convert the LabelMe JSON annotations to YOLO format for training.

Run the following command:

```
python convert_labelme_to_yolo.py --json_dir data\two_stage\powerline\labeled\images --output_dir data\two_stage\powerline\labeled\labels
```

This script will:

1. Process all JSON files in the images directory
2. Convert the annotations to YOLO format (class_id x_center y_center width height)
3. Save the results as .txt files in the labels directory

## 4. Continuing the Pipeline

Once your annotations are converted to YOLO format, you can continue with the next step in your pipeline:

```
python run.py two-stage --mode augment-powerlines --images data/two_stage/powerline/labeled --output data/two_stage/powerline/augmented
```

## Important Tips

- **Be Consistent**: Draw boxes that consistently capture the same parts of the powerlines
- **Handle Occlusions**: For partially occluded powerlines, annotate the visible parts
- **Quality Over Quantity**: It's better to have fewer, high-quality annotations than many poor ones
- **Save Progress Regularly**: Although LabelMe autosaves, it's good practice to check that files are being saved correctly

## Troubleshooting

- If LabelMe crashes, your annotations should still be saved as JSON files
- If you have issues with the conversion script, check that your JSON files are in the correct format
- Make sure your labels match exactly what's in your classes.txt file
