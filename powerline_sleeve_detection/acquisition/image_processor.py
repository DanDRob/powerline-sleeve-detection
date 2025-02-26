import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, List, Tuple, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import time

from ..system.config import Config
from ..system.logging import get_logger


class ImageProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("image_processor")

    def preprocess_image(self, image: Image.Image, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Preprocess an image for improved sleeve detection.

        Args:
            image: PIL Image object
            params: Optional parameters for preprocessing

        Returns:
            Dictionary with processed image and metadata
        """
        if image is None:
            self.logger.error("Cannot process None image")
            return {"success": False, "error": "Image is None"}

        try:
            # Convert to numpy array (RGB)
            img_array = np.array(image)

            # Convert to BGR for OpenCV
            img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Apply image enhancements
            processed_cv = self._enhance_image(img_cv)

            # Convert back to RGB for PIL
            processed_rgb = cv2.cvtColor(processed_cv, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            processed_image = Image.fromarray(processed_rgb)

            return {
                "original_image": image,
                "processed_image": processed_image,
                "params": params or {},
                "success": True
            }

        except Exception as e:
            self.logger.error(f"Image preprocessing failed: {e}")
            return {
                "original_image": image,
                "processed_image": None,
                "params": params or {},
                "success": False,
                "error": f"Image preprocessing failed: {e}"
            }

    def _enhance_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply various enhancements to improve detection.

        Args:
            img: OpenCV image in BGR format

        Returns:
            Enhanced image
        """
        # Convert to grayscale for certain operations
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply contrast normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(gray)

        # Edge enhancement
        edges = cv2.Canny(cl1, 50, 150)

        # Combine edge information with original image
        edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.addWeighted(img, 0.7, edge_colored, 0.3, 0)

        # Apply denoising
        denoised = cv2.fastNlMeansDenoisingColored(
            enhanced, None, 10, 10, 7, 21)

        return denoised

    def create_augmented_images(self, image: Image.Image, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Create augmented versions of an image for training or detection.

        Args:
            image: PIL Image object
            params: Optional parameters for augmentation

        Returns:
            List of dictionaries with augmented images and metadata
        """
        if image is None:
            self.logger.error("Cannot augment None image")
            return []

        augmented_images = []
        params = params or {}

        try:
            # Base augmentations
            # 1. Brightness variations
            for brightness_factor in [0.7, 1.3]:
                brightness_img = ImageEnhance.Brightness(
                    image).enhance(brightness_factor)
                augmented_images.append({
                    "image": brightness_img,
                    "augmentation_type": "brightness",
                    "factor": brightness_factor,
                    "params": params.copy(),
                    "success": True
                })

            # 2. Contrast variations
            for contrast_factor in [0.7, 1.3]:
                contrast_img = ImageEnhance.Contrast(
                    image).enhance(contrast_factor)
                augmented_images.append({
                    "image": contrast_img,
                    "augmentation_type": "contrast",
                    "factor": contrast_factor,
                    "params": params.copy(),
                    "success": True
                })

            # 3. Blur for simulating different weather/focus conditions
            blur_img = image.filter(ImageFilter.GaussianBlur(radius=1.5))
            augmented_images.append({
                "image": blur_img,
                "augmentation_type": "blur",
                "factor": 1.5,
                "params": params.copy(),
                "success": True
            })

            # 4. Color modifications (simulating different lighting)
            for color_factor in [0.8, 1.2]:
                color_img = ImageEnhance.Color(image).enhance(color_factor)
                augmented_images.append({
                    "image": color_img,
                    "augmentation_type": "color",
                    "factor": color_factor,
                    "params": params.copy(),
                    "success": True
                })

            return augmented_images

        except Exception as e:
            self.logger.error(f"Image augmentation failed: {e}")
            return []

    def batch_process_images(self, images: List[Dict[str, Any]],
                             with_augmentation: bool = False) -> Dict[str, Any]:
        """
        Process a batch of images using parallel processing.

        Args:
            images: List of image dictionaries (from StreetViewClient)
            with_augmentation: Whether to create augmented versions

        Returns:
            Dictionary with processed images and metadata
        """
        self.logger.info(f"Batch processing {len(images)} images")
        start_time = time.time()

        successful_images = [img for img in images if img.get(
            "success", False) and img.get("image") is not None]
        failed_images = [img for img in images if not img.get(
            "success", False) or img.get("image") is None]

        processed_results = []
        augmented_results = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.config.system.num_workers) as executor:
            # Submit preprocessing tasks
            preprocess_futures = [
                executor.submit(self.preprocess_image,
                                img["image"], img.get("params", {}))
                for img in successful_images
            ]

            # Process results as they complete
            for future in preprocess_futures:
                try:
                    result = future.result()
                    if result.get("success", False):
                        processed_results.append(result)

                        # If augmentation requested, process augmented versions
                        if with_augmentation:
                            augmented_images = self.create_augmented_images(
                                result["processed_image"],
                                result.get("params", {})
                            )
                            augmented_results.extend(augmented_images)
                    else:
                        failed_images.append(result)
                except Exception as e:
                    self.logger.error(f"Processing task failed: {e}")

        elapsed_time = time.time() - start_time
        self.logger.info(f"Batch processing completed in {elapsed_time:.2f}s. "
                         f"Processed: {len(processed_results)}, "
                         f"Augmented: {len(augmented_results)}, "
                         f"Failed: {len(failed_images)}")

        return {
            "processed_images": processed_results,
            "augmented_images": augmented_results,
            "failed_images": failed_images,
            "metadata": {
                "total_input": len(images),
                "total_processed": len(processed_results),
                "total_augmented": len(augmented_results),
                "total_failed": len(failed_images),
                "processing_time": elapsed_time
            },
            "success": True
        }

    def save_processed_images(self, processing_results: Dict[str, Any],
                              output_dir: str = None) -> Dict[str, Any]:
        """
        Save processed images to disk.

        Args:
            processing_results: Results from batch_process_images
            output_dir: Directory to save images (uses config if None)

        Returns:
            Dictionary with file paths and metadata
        """
        if output_dir is None:
            output_dir = os.path.join(
                self.config.system.output_dir, "processed_images")

        # Create output directories
        processed_dir = os.path.join(output_dir, "processed")
        augmented_dir = os.path.join(output_dir, "augmented")
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(augmented_dir, exist_ok=True)

        saved_files = {
            "processed": [],
            "augmented": []
        }

        # Save processed images
        for i, result in enumerate(processing_results.get("processed_images", [])):
            if not result.get("success", False) or result.get("processed_image") is None:
                continue

            # Extract parameters
            params = result.get("params", {})
            point_index = params.get("point_index", i)
            heading = params.get("heading", 0)
            pitch = params.get("pitch", 0)
            fov = params.get("fov", 55)

            # Determine side (right or left)
            road_heading = params.get("road_heading", 0)
            relative_heading = (heading - road_heading) % 360
            side = "right" if relative_heading in [45, 90, 135] else "left"

            # Create filename
            filename = f"processed_{side}_{point_index}_heading_{int(heading)}_pitch_{int(pitch)}_fov_{int(fov)}.jpg"
            filepath = os.path.join(processed_dir, filename)

            # Save image
            try:
                result["processed_image"].save(filepath)
                saved_files["processed"].append({
                    "filepath": filepath,
                    "filename": filename,
                    "params": params,
                    "side": side
                })
                self.logger.debug(f"Saved processed image: {filename}")
            except Exception as e:
                self.logger.error(
                    f"Failed to save processed image {filename}: {e}")

        # Save augmented images
        for i, result in enumerate(processing_results.get("augmented_images", [])):
            if not result.get("success", False) or result.get("image") is None:
                continue

            # Extract parameters
            params = result.get("params", {})
            point_index = params.get("point_index", i)
            heading = params.get("heading", 0)
            pitch = params.get("pitch", 0)
            fov = params.get("fov", 55)
            augmentation_type = result.get("augmentation_type", "unknown")
            factor = result.get("factor", 1.0)

            # Determine side (right or left)
            road_heading = params.get("road_heading", 0)
            relative_heading = (heading - road_heading) % 360
            side = "right" if relative_heading in [45, 90, 135] else "left"

            # Create filename
            filename = f"augmented_{side}_{point_index}_{augmentation_type}_{factor:.1f}_heading_{int(heading)}_pitch_{int(pitch)}_fov_{int(fov)}.jpg"
            filepath = os.path.join(augmented_dir, filename)

            # Save image
            try:
                result["image"].save(filepath)
                saved_files["augmented"].append({
                    "filepath": filepath,
                    "filename": filename,
                    "params": params,
                    "side": side,
                    "augmentation_type": augmentation_type,
                    "factor": factor
                })
                self.logger.debug(f"Saved augmented image: {filename}")
            except Exception as e:
                self.logger.error(
                    f"Failed to save augmented image {filename}: {e}")

        total_saved = len(saved_files["processed"]) + \
            len(saved_files["augmented"])
        self.logger.info(f"Saved {total_saved} images to {output_dir} "
                         f"(Processed: {len(saved_files['processed'])}, "
                         f"Augmented: {len(saved_files['augmented'])})")

        return {
            "success": True,
            "saved_files": saved_files,
            "metadata": {
                "total_saved": total_saved,
                "processed_saved": len(saved_files["processed"]),
                "augmented_saved": len(saved_files["augmented"]),
                "output_dir": output_dir,
                "processed_dir": processed_dir,
                "augmented_dir": augmented_dir
            }
        }
