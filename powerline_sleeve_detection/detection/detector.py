import os
import time
import numpy as np
import torch
from PIL import Image, ImageDraw
import cv2
from typing import Dict, List, Tuple, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import gc

from ..system.config import Config
from ..system.logging import get_logger
from .model_manager import ModelManager
from .ensemble_integration import EnsembleIntegration


class SleeveDetector:
    def __init__(self, config: Config, model_manager: ModelManager = None):
        self.config = config
        self.logger = get_logger("sleeve_detector")
        self.model_manager = model_manager or ModelManager(config)
        self.device = self.model_manager.device

        # Register models but don't load them yet (lazy loading)
        if not self.model_manager.models:
            self.model_manager.register_models()

        # Track current batch size
        self.current_batch_size = config.detection.batch_size
        self.memory_errors = 0
        self.last_memory_check = time.time()

        # Initialize ensemble integration if enabled
        self.ensemble_integration = None
        if config.get('ensemble.enabled', False):
            try:
                self.ensemble_integration = EnsembleIntegration(
                    config, self.model_manager)
                self.ensemble_integration.integrate_with_detector(self)
                self.logger.info(
                    "Ensemble integration initialized and integrated with detector")
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize ensemble integration: {e}")

    def detect(self, image: Image.Image, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect sleeves in a single image using a specific model.

        Args:
            image: PIL Image
            model_name: Name of the model to use (None for default)

        Returns:
            Dictionary with detection results
        """
        if image is None:
            self.logger.error("Cannot detect on None image")
            return {"success": False, "error": "Image is None"}

        start_time = time.time()

        try:
            # Get the model (will be loaded if not already in memory)
            if model_name:
                model = self.model_manager.get_model(model_name)
                if model is None:
                    self.logger.warning(
                        f"Model {model_name} not found, using first available model")
                    ensemble_models = self.model_manager.get_ensemble_models()
                    if not ensemble_models:
                        return {"success": False, "error": "No models available"}
                    model = ensemble_models[0]["model"]
                    model_name = ensemble_models[0]["name"]
            else:
                ensemble_models = self.model_manager.get_ensemble_models()
                if not ensemble_models:
                    return {"success": False, "error": "No models available"}
                model = ensemble_models[0]["model"]
                model_name = ensemble_models[0]["name"]

            # Get model type
            model_type = self.model_manager.models[model_name]["type"]

            # Run detection based on model type
            if model_type == "yolov5":
                results = self._detect_yolov5(model, image)
            elif model_type == "yolov8":
                results = self._detect_yolov8(model, image)
            elif model_type == "efficientdet":
                results = self._detect_efficientdet(model, image)
            else:
                self.logger.error(f"Unsupported model type: {model_type}")
                return {"success": False, "error": f"Unsupported model type: {model_type}"}

            # Post-process results
            processed_results = self._process_results(
                results, model_type, image.size)

            elapsed_time = time.time() - start_time

            return {
                "detections": processed_results,
                "model_name": model_name,
                "model_type": model_type,
                "success": True,
                "processing_time": elapsed_time,
                "image_size": image.size
            }

        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return {"success": False, "error": str(e)}

    def detect_ensemble(self, image: Image.Image) -> Dict[str, Any]:
        """
        Detect sleeves using an ensemble of models.

        Args:
            image: PIL Image

        Returns:
            Dictionary with ensemble detection results
        """
        if image is None:
            self.logger.error("Cannot detect on None image")
            return {"success": False, "error": "Image is None"}

        start_time = time.time()

        try:
            ensemble_models = self.model_manager.get_ensemble_models()

            if not ensemble_models:
                self.logger.error("No models available for ensemble detection")
                return {"success": False, "error": "No models available"}

            # Run detection with each model
            model_results = []
            for model_info in ensemble_models:
                model_name = model_info["name"]
                self.logger.debug(f"Running detection with model {model_name}")
                result = self.detect(image, model_name)
                if result["success"]:
                    model_results.append({
                        "result": result,
                        "weight": model_info["weight"]
                    })

            # Combine results using weighted box fusion
            ensemble_detections = self._ensemble_detections([r["result"]["detections"] for r in model_results],
                                                            [r["weight"] for r in model_results])

            elapsed_time = time.time() - start_time

            return {
                "detections": ensemble_detections,
                "model_results": model_results,
                "success": True,
                "processing_time": elapsed_time,
                "image_size": image.size,
                "ensemble_size": len(model_results)
            }

        except Exception as e:
            self.logger.error(f"Ensemble detection failed: {e}")
            return {"success": False, "error": str(e)}

    def _detect_yolov5(self, model: Any, image: Image.Image) -> Any:
        """Run detection with YOLOv5 model"""
        # YOLOv5 accepts PIL images directly
        results = model(image)
        return results

    def _detect_yolov8(self, model: Any, image: Image.Image) -> Any:
        """Run detection with YOLOv8 model"""
        # YOLOv8 accepts PIL images directly
        results = model(image)
        return results

    def _detect_efficientdet(self, model: Any, image: Image.Image) -> Any:
        """Run detection with EfficientDet model"""
        # Convert PIL image to tensor
        import torch.nn.functional as F

        img_tensor = torch.ByteTensor(
            torch.ByteStorage.from_buffer(image.tobytes()))
        img_tensor = img_tensor.view(
            image.size[1], image.size[0], len(image.getbands()))
        img_tensor = img_tensor.permute((2, 0, 1)).float().div(255)

        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)

        # Resize to expected input size
        img_tensor = F.interpolate(img_tensor, size=(
            512, 512), mode='bilinear', align_corners=False)

        # Move to device
        img_tensor = img_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            results = model(img_tensor)

        return results

    def _process_results(self, results: Any, model_type: str, image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Process raw detection results into a standardized format"""
        detections = []
        width, height = image_size
        confidence_threshold = self.config.detection.confidence_threshold

        try:
            if model_type == "yolov5":
                # Extract YOLOv5 results
                result_tensor = results.xyxy[0]  # First image, xyxy format

                for *xyxy, conf, cls in result_tensor.cpu().numpy():
                    if conf >= confidence_threshold:
                        x1, y1, x2, y2 = map(float, xyxy)

                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": float(conf),
                            "class_id": int(cls),
                            "normalized_bbox": [x1/width, y1/height, x2/width, y2/height]
                        })

            elif model_type == "yolov8":
                # Extract YOLOv8 results (different API)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        if box.conf >= confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                            detections.append({
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "confidence": float(box.conf),
                                "class_id": int(box.cls),
                                "normalized_bbox": [float(x1/width), float(y1/height),
                                                    float(x2/width), float(y2/height)]
                            })

            elif model_type == "efficientdet":
                # Process EfficientDet results
                # This is a simplified example, as the actual output format depends on implementation
                boxes = results[0].detach().cpu().numpy()
                scores = results[1].detach().cpu().numpy()
                labels = results[2].detach().cpu().numpy()

                for i in range(len(boxes)):
                    if scores[i] >= confidence_threshold:
                        # Rescale box coordinates to original image
                        x1, y1, x2, y2 = boxes[i] * \
                            np.array([width, height, width, height]) / 512

                        detections.append({
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": float(scores[i]),
                            "class_id": int(labels[i]),
                            "normalized_bbox": [float(x1/width), float(y1/height),
                                                float(x2/width), float(y2/height)]
                        })

        except Exception as e:
            self.logger.error(f"Error processing {model_type} results: {e}")

        return detections

    def _ensemble_detections(self, all_detections: List[List[Dict[str, Any]]],
                             weights: List[float]) -> List[Dict[str, Any]]:
        """
        Combine detections from multiple models using weighted box fusion.

        Args:
            all_detections: List of detection lists from each model
            weights: List of weights for each model

        Returns:
            Combined detection list
        """
        if not all_detections:
            return []

        # Simple approach: combine all boxes, then apply NMS
        combined_detections = []
        for i, detections in enumerate(all_detections):
            for det in detections:
                # Add model weight
                det["weighted_confidence"] = det["confidence"] * weights[i]
                combined_detections.append(det)

        # Apply NMS on combined detections
        final_detections = self._apply_nms(
            combined_detections, self.config.detection.iou_threshold)

        # Sort by confidence
        final_detections.sort(key=lambda x: x["confidence"], reverse=True)

        return final_detections

    def _apply_nms(self, detections: List[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
        """
        Apply non-maximum suppression to remove overlapping detections.

        Args:
            detections: List of detections
            iou_threshold: IoU threshold for suppression

        Returns:
            Filtered detections
        """
        if not detections:
            return []

        # Convert to numpy arrays for easier processing
        boxes = np.array([d["bbox"] for d in detections])
        scores = np.array(
            [d.get("weighted_confidence", d["confidence"]) for d in detections])

        # Compute areas
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # Sort by confidence
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Compute IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        # Keep selected detections
        filtered_detections = [detections[i] for i in keep]

        # Update confidence to weighted confidence if available
        for det in filtered_detections:
            if "weighted_confidence" in det:
                det["confidence"] = det["weighted_confidence"]
                del det["weighted_confidence"]

        return filtered_detections

    def _estimate_optimal_batch_size(self, current_batch_size: int) -> int:
        """
        Estimate optimal batch size based on memory usage and previous errors.

        Args:
            current_batch_size: Current batch size

        Returns:
            New batch size to use
        """
        # If we've had memory errors, reduce batch size
        if self.memory_errors > 0:
            # Reset error count after reducing
            self.memory_errors = 0
            new_size = max(1, current_batch_size // 2)
            self.logger.info(
                f"Reducing batch size from {current_batch_size} to {new_size} due to memory errors")
            return new_size

        # If it's been a while since we checked memory and haven't had errors, try increasing
        if time.time() - self.last_memory_check > 300:  # Every 5 minutes
            self.last_memory_check = time.time()

            # Only increase if using CUDA
            if torch.cuda.is_available():
                # Check available memory
                free_memory = torch.cuda.get_device_properties(
                    0).total_memory - torch.cuda.memory_allocated()
                free_memory_mb = free_memory / (1024 * 1024)

                # If we have plenty of free memory, cautiously increase batch size
                if free_memory_mb > 2000:  # 2GB free
                    new_size = min(current_batch_size + 1,
                                   self.config.detection.batch_size * 2)
                    if new_size > current_batch_size:
                        self.logger.info(
                            f"Increasing batch size from {current_batch_size} to {new_size} (free memory: {free_memory_mb:.0f}MB)")
                        return new_size

        # Otherwise keep current size
        return current_batch_size

    def batch_detect(self, images: List[Dict[str, Any]], use_ensemble: bool = True,
                     parallel: bool = True) -> Dict[str, Any]:
        """
        Run detection on a batch of images.

        Args:
            images: List of image dictionaries (each with 'image' key)
            use_ensemble: Whether to use ensemble detection
            parallel: Whether to process in parallel

        Returns:
            Dictionary with detection results
        """
        self.logger.info(
            f"Batch detecting on {len(images)} images (ensemble={use_ensemble}, parallel={parallel})")
        start_time = time.time()

        valid_images = [
            img for img in images if "image" in img and img["image"] is not None]
        invalid_images = [
            img for img in images if "image" not in img or img["image"] is None]

        detection_results = []

        # Try to optimize batch processing based on available memory
        batch_size = self._estimate_optimal_batch_size(self.current_batch_size)
        self.current_batch_size = batch_size

        if parallel and len(valid_images) > 1:
            # Process in batches to avoid memory issues
            for i in range(0, len(valid_images), batch_size):
                batch = valid_images[i:i+batch_size]
                self.logger.debug(
                    f"Processing batch {i//batch_size + 1}/{(len(valid_images)-1)//batch_size + 1} with {len(batch)} images")

                try:
                    # Use ThreadPoolExecutor for parallel processing
                    with ThreadPoolExecutor(max_workers=min(self.config.system.num_workers, len(batch))) as executor:
                        # Submit detection tasks
                        futures = []
                        for img_dict in batch:
                            if use_ensemble:
                                future = executor.submit(
                                    self.detect_ensemble, img_dict["image"])
                            else:
                                future = executor.submit(
                                    self.detect, img_dict["image"])
                            futures.append((future, img_dict))

                        # Process results as they complete
                        for future, img_dict in futures:
                            try:
                                result = future.result()
                                if result["success"]:
                                    detection_results.append({
                                        "image_params": img_dict.get("params", {}),
                                        "detection_result": result
                                    })
                                else:
                                    invalid_images.append(img_dict)
                            except Exception as e:
                                self.logger.error(
                                    f"Detection task failed: {e}")
                                invalid_images.append(img_dict)

                    # Clean up after batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    # Handle out-of-memory errors
                    if "CUDA out of memory" in str(e):
                        self.memory_errors += 1
                        self.logger.warning(f"CUDA out of memory error: {e}")

                        # Reduce batch size for next attempt
                        batch_size = max(1, batch_size // 2)
                        self.current_batch_size = batch_size
                        self.logger.info(
                            f"Reduced batch size to {batch_size} due to memory error")

                        # Fall back to sequential processing for this batch
                        for img_dict in batch:
                            try:
                                if use_ensemble:
                                    result = self.detect_ensemble(
                                        img_dict["image"])
                                else:
                                    result = self.detect(img_dict["image"])

                                if result["success"]:
                                    detection_results.append({
                                        "image_params": img_dict.get("params", {}),
                                        "detection_result": result
                                    })
                                else:
                                    invalid_images.append(img_dict)
                            except Exception as nested_e:
                                self.logger.error(
                                    f"Sequential fallback detection failed: {nested_e}")
                                invalid_images.append(img_dict)
                    else:
                        # Re-raise other runtime errors
                        raise
        else:
            # Sequential processing
            for img_dict in valid_images:
                try:
                    if use_ensemble:
                        result = self.detect_ensemble(img_dict["image"])
                    else:
                        result = self.detect(img_dict["image"])

                    if result["success"]:
                        detection_results.append({
                            "image_params": img_dict.get("params", {}),
                            "detection_result": result
                        })
                    else:
                        invalid_images.append(img_dict)
                except Exception as e:
                    self.logger.error(f"Detection failed: {e}")
                    invalid_images.append(img_dict)

        elapsed_time = time.time() - start_time

        # Count detected sleeves
        total_detections = sum(
            len(r["detection_result"]["detections"]) for r in detection_results)
        successful_images = len(detection_results)

        self.logger.info(f"Batch detection completed in {elapsed_time:.2f}s. "
                         f"Found {total_detections} detections in {successful_images}/{len(images)} images.")

        # Get status of models after batch is complete
        model_status = self.model_manager.get_model_status()

        return {
            "detection_results": detection_results,
            "failed_images": invalid_images,
            "success": True,
            "metadata": {
                "total_images": len(images),
                "successful_images": successful_images,
                "failed_images": len(invalid_images),
                "total_detections": total_detections,
                "processing_time": elapsed_time,
                "batch_size": self.current_batch_size,
                "ensemble": use_ensemble,
                "parallel": parallel,
                "model_status": model_status
            }
        }

    def draw_detections(self, image: Image.Image, detections: List[Dict[str, Any]],
                        output_path: Optional[str] = None) -> Image.Image:
        """
        Draw detection boxes on an image.

        Args:
            image: PIL Image
            detections: List of detection dictionaries
            output_path: Optional path to save the output image

        Returns:
            PIL Image with drawn detections
        """
        if image is None or not detections:
            return image

        # Create a copy to avoid modifying the original
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)

        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Draw confidence label
            draw.text([x1, y1-15], f"Sleeve: {conf:.2f}", fill="red")

        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img_draw.save(output_path)
            self.logger.debug(
                f"Saved detection visualization to {output_path}")

        return img_draw

    def save_detection_results(self, batch_results: Dict[str, Any],
                               output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Save detection results and visualization images.

        Args:
            batch_results: Results from batch_detect
            output_dir: Directory to save results (uses config if None)

        Returns:
            Dictionary with saved file information
        """
        if output_dir is None:
            output_dir = os.path.join(
                self.config.system.output_dir, "detections")

        # Create output directories
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        saved_files = []

        for i, result in enumerate(batch_results.get("detection_results", [])):
            detection_result = result["detection_result"]
            params = result["image_params"]

            # Extract parameters for filename
            point_index = params.get("point_index", i)
            heading = params.get("heading", 0)
            pitch = params.get("pitch", 0)
            fov = params.get("fov", 55)

            # Determine side (right or left)
            road_heading = params.get("road_heading", 0)
            relative_heading = (heading - road_heading) % 360
            side = "right" if relative_heading in [45, 90, 135] else "left"

            # Create filename for the visualization
            has_detections = len(detection_result["detections"]) > 0
            detection_status = "positive" if has_detections else "negative"

            filename = f"detection_{detection_status}_{side}_{point_index}_heading_{int(heading)}_pitch_{int(pitch)}_fov_{int(fov)}.jpg"
            filepath = os.path.join(images_dir, filename)

            # Draw detections on the image
            if "original_image" in result:
                image = result["original_image"]
            else:
                # Get image from processed results if available
                processed_images = batch_results.get("processed_images", [])
                image = None
                for processed in processed_images:
                    if processed.get("params", {}).get("point_index") == point_index:
                        image = processed.get("original_image")
                        break

            if image is not None:
                # Draw and save visualization
                self.draw_detections(
                    image, detection_result["detections"], filepath)

                saved_files.append({
                    "filepath": filepath,
                    "filename": filename,
                    "params": params,
                    "detections": len(detection_result["detections"]),
                    "has_detections": has_detections
                })

        self.logger.info(
            f"Saved {len(saved_files)} detection visualizations to {output_dir}")

        return {
            "success": True,
            "saved_files": saved_files,
            "metadata": {
                "total_saved": len(saved_files),
                "output_dir": output_dir,
                "images_dir": images_dir
            }
        }
