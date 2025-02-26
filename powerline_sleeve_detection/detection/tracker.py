import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import time
import uuid
from dataclasses import dataclass, field

from ..system.config import Config
from ..system.logging import get_logger


@dataclass
class TrackedObject:
    object_id: str
    class_id: int
    first_seen: float
    last_seen: float
    positions: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    age: int = 1
    max_disappeared: int = 5
    disappeared: int = 0

    def update(self, detection: Dict[str, Any], timestamp: float) -> None:
        """Update tracked object with new detection"""
        self.last_seen = timestamp
        self.positions.append(detection)
        self.confidence = max(self.confidence, detection["confidence"])
        self.age += 1
        self.disappeared = 0

    def mark_disappeared(self) -> bool:
        """Mark object as disappeared and return whether it should be removed"""
        self.disappeared += 1
        return self.disappeared > self.max_disappeared


class SleevePowerlineTracker:
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("sleeve_tracker")
        self.tracked_objects = {}  # Dictionary of tracked objects by ID
        self.last_object_id = 0  # Counter for object IDs

        # Parameters
        self.min_confidence = config.detection.confidence_threshold
        self.iou_threshold = 0.3  # Threshold for matching detections to tracks
        self.max_disappeared = 5  # Maximum number of frames an object can disappear

        # Store geo-referenced detections
        self.geo_referenced_detections = []

        # Performance tracking
        self.start_time = time.time()

    def _calculate_iou(self, boxA: List[float], boxB: List[float]) -> float:
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Args:
            boxA: First bounding box [x1, y1, x2, y2]
            boxB: Second bounding box [x1, y1, x2, y2]

        Returns:
            IoU value
        """
        # Calculate intersection coordinates
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Calculate area of intersection
        intersection_area = max(0, xB - xA) * max(0, yB - yA)

        # Calculate area of both bounding boxes
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Calculate IoU
        iou = intersection_area / \
            float(boxA_area + boxB_area - intersection_area)

        return iou

    def _generate_object_id(self) -> str:
        """Generate a unique object ID"""
        return str(uuid.uuid4())

    def update(self, detections: List[Dict[str, Any]], image_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detection dictionaries
            image_params: Parameters of the image (location, etc.)

        Returns:
            Dictionary with tracking results
        """
        timestamp = time.time()
        current_objects = {}
        unmatched_detections = []
        matched_detections = []

        self.logger.debug(
            f"Updating tracker with {len(detections)} detections")

        # Filter detections based on confidence
        filtered_detections = [
            d for d in detections if d["confidence"] >= self.min_confidence]

        # If we have existing tracked objects
        if self.tracked_objects:
            # Calculate IoU between existing objects and new detections
            iou_matrix = np.zeros(
                (len(self.tracked_objects), len(filtered_detections)))

            for i, (object_id, tracked_obj) in enumerate(self.tracked_objects.items()):
                if tracked_obj.positions:
                    last_box = tracked_obj.positions[-1]["bbox"]
                    for j, detection in enumerate(filtered_detections):
                        iou_matrix[i, j] = self._calculate_iou(
                            last_box, detection["bbox"])

            # Match detections to tracked objects
            if iou_matrix.size > 0:
                # Find max IoU for each object/detection pair
                matched_indices = []

                # For each tracked object
                for i in range(iou_matrix.shape[0]):
                    # Get indices of all detections that match this object
                    matches = np.where(
                        iou_matrix[i, :] >= self.iou_threshold)[0]

                    if len(matches) > 0:
                        # Get the detection with highest IoU
                        j = matches[np.argmax(iou_matrix[i, matches])]
                        matched_indices.append((i, j))
                        matched_detections.append(j)

                # Update matched objects
                for i, j in matched_indices:
                    object_id = list(self.tracked_objects.keys())[i]
                    tracked_obj = self.tracked_objects[object_id]
                    tracked_obj.update(filtered_detections[j], timestamp)
                    current_objects[object_id] = tracked_obj

                    # Add geolocation data to this detection
                    geo_detection = {
                        **filtered_detections[j],
                        "object_id": object_id,
                        "image_params": image_params,
                        "timestamp": timestamp,
                        "age": tracked_obj.age
                    }
                    self.geo_referenced_detections.append(geo_detection)

            # Find unmatched detections
            unmatched_detections = [j for j in range(
                len(filtered_detections)) if j not in matched_detections]
        else:
            # All detections are unmatched if there are no existing tracked objects
            unmatched_detections = list(range(len(filtered_detections)))

        # Create new tracked objects for unmatched detections
        for j in unmatched_detections:
            object_id = self._generate_object_id()
            new_obj = TrackedObject(
                object_id=object_id,
                class_id=filtered_detections[j].get("class_id", 0),
                first_seen=timestamp,
                last_seen=timestamp,
                confidence=filtered_detections[j]["confidence"],
                max_disappeared=self.max_disappeared
            )
            new_obj.update(filtered_detections[j], timestamp)
            current_objects[object_id] = new_obj

            # Add geolocation data to this detection
            geo_detection = {
                **filtered_detections[j],
                "object_id": object_id,
                "image_params": image_params,
                "timestamp": timestamp,
                "age": 1
            }
            self.geo_referenced_detections.append(geo_detection)

        # Handle disappeared objects
        for object_id, tracked_obj in self.tracked_objects.items():
            if object_id not in current_objects:
                # Mark as disappeared
                should_remove = tracked_obj.mark_disappeared()
                if not should_remove:
                    current_objects[object_id] = tracked_obj

        # Update tracked objects
        self.tracked_objects = current_objects

        # Return results
        return {
            "tracked_objects": self.tracked_objects,
            "num_objects": len(self.tracked_objects),
            "matched_detections": len(matched_detections),
            "new_detections": len(unmatched_detections),
            "timestamp": timestamp
        }

    def process_batch_results(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process batch detection results and update tracker.

        Args:
            batch_results: Results from detector.batch_detect()

        Returns:
            Dictionary with tracking results
        """
        tracking_results = []
        start_time = time.time()

        for result in batch_results.get("detection_results", []):
            detection_result = result["detection_result"]
            image_params = result["image_params"]

            # Update tracker with these detections
            track_update = self.update(
                detection_result["detections"], image_params)

            tracking_results.append({
                "image_params": image_params,
                "tracking_result": track_update,
                "num_detections": len(detection_result["detections"])
            })

        elapsed_time = time.time() - start_time

        self.logger.info(f"Processed {len(tracking_results)} results in {elapsed_time:.2f}s. "
                         f"Currently tracking {len(self.tracked_objects)} unique objects.")

        return {
            "tracking_results": tracking_results,
            "tracked_objects": self.tracked_objects,
            "success": True,
            "metadata": {
                "total_images": len(tracking_results),
                "unique_objects": len(self.tracked_objects),
                "geo_referenced_detections": len(self.geo_referenced_detections),
                "processing_time": elapsed_time
            }
        }

    def get_geo_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Convert geo-referenced detections to a GeoDataFrame.

        Returns:
            GeoDataFrame with detection data and coordinates
        """
        if not self.geo_referenced_detections:
            return None

        try:
            import geopandas as gpd
            from shapely.geometry import Point

            # Create dataframe from detections
            df = pd.DataFrame(self.geo_referenced_detections)

            # Extract coordinates
            geometries = []
            for _, row in df.iterrows():
                lat = row["image_params"].get("latitude")
                lng = row["image_params"].get("longitude")
                if lat is not None and lng is not None:
                    geometries.append(Point(lng, lat))
                else:
                    geometries.append(None)

            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")

            self.logger.info(
                f"Created GeoDataFrame with {len(gdf)} detections")
            return gdf

        except ImportError:
            self.logger.warning(
                "Could not create GeoDataFrame: geopandas not installed")
            return pd.DataFrame(self.geo_referenced_detections)
        except Exception as e:
            self.logger.error(f"Error creating GeoDataFrame: {e}")
            return pd.DataFrame(self.geo_referenced_detections)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about tracked objects.

        Returns:
            Dictionary with tracking statistics
        """
        # Count objects by class
        class_counts = defaultdict(int)
        for obj_id, obj in self.tracked_objects.items():
            class_counts[obj.class_id] += 1

        # Calculate confidence stats
        if self.tracked_objects:
            confidences = [
                obj.confidence for obj in self.tracked_objects.values()]
            avg_confidence = sum(confidences) / len(confidences)
            max_confidence = max(confidences)
            min_confidence = min(confidences)
        else:
            avg_confidence = 0
            max_confidence = 0
            min_confidence = 0

        # Calculate tracking duration
        tracking_duration = time.time() - self.start_time

        return {
            "total_objects": len(self.tracked_objects),
            "total_detections": len(self.geo_referenced_detections),
            "class_counts": dict(class_counts),
            "confidence_stats": {
                "average": avg_confidence,
                "max": max_confidence,
                "min": min_confidence
            },
            "tracking_duration": tracking_duration
        }

    def save_to_csv(self, output_path: str) -> str:
        """
        Save geo-referenced detections to CSV.

        Args:
            output_path: Path to save CSV file

        Returns:
            Path to saved file
        """
        if not self.geo_referenced_detections:
            self.logger.warning("No detections to save")
            return None

        try:
            # Flatten nested dictionaries
            flattened_detections = []
            for det in self.geo_referenced_detections:
                flat_det = {
                    "object_id": det["object_id"],
                    "confidence": det["confidence"],
                    "class_id": det.get("class_id", 0),
                    "timestamp": det["timestamp"],
                    "age": det["age"],
                    "bbox_x1": det["bbox"][0],
                    "bbox_y1": det["bbox"][1],
                    "bbox_x2": det["bbox"][2],
                    "bbox_y2": det["bbox"][3]
                }

                # Add image parameters
                for k, v in det["image_params"].items():
                    flat_det[f"image_{k}"] = v

                flattened_detections.append(flat_det)

            df = pd.DataFrame(flattened_detections)
            df.to_csv(output_path, index=False)

            self.logger.info(f"Saved {len(df)} detections to {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error saving detections to CSV: {e}")
            return None
