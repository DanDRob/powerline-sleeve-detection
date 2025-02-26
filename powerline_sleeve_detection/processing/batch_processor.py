import os
import json
import time
import asyncio
import csv
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import concurrent.futures
from datetime import datetime
import logging

from ..system.config import Config
from ..system.logging import get_logger
from ..acquisition.route_planner import RoutePlanner
from ..acquisition.streetview_client import StreetViewClient
from ..acquisition.image_processor import ImageProcessor
from ..detection.detector import SleeveDetector
from ..detection.model_manager import ModelManager
from ..detection.tracker import SleevePowerlineTracker
from ..visualization.map_generator import MapGenerator
from ..acquisition.cache_manager import CacheManager


class BatchProcessor:
    """
    Process multiple routes across Ontario to detect powerline sleeves.
    Handles route planning, image acquisition, detection, and visualization.
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("batch_processor")

        # Initialize components
        self.route_planner = RoutePlanner(config)
        self.model_manager = ModelManager(config)
        self.detector = SleeveDetector(config, self.model_manager)
        self.tracker = SleevePowerlineTracker(config)
        self.map_generator = MapGenerator(config)
        self.cache_manager = CacheManager(config)

        # Create output directories
        self.output_dir = os.path.join(
            config.system.output_dir, "batch_results")
        os.makedirs(self.output_dir, exist_ok=True)

        # Tracking for processed routes
        self.routes_processed = 0
        self.detection_results = {}
        self.start_time = time.time()

    async def process_route(self, route_id: str, start_location: str, end_location: str,
                            save_images: bool = True, draw_detections: bool = True,
                            generate_map: bool = True) -> Dict[str, Any]:
        """
        Process a single route from start to end.

        Args:
            route_id: Unique identifier for the route
            start_location: Starting location (address or lat,lng)
            end_location: Ending location (address or lat,lng)
            save_images: Whether to save acquired images to disk
            draw_detections: Whether to draw and save detection visualizations
            generate_map: Whether to generate a coverage map

        Returns:
            Dictionary with processing results
        """
        # Create a directory for this route
        route_dir = os.path.join(self.output_dir, f"route_{route_id}")
        os.makedirs(route_dir, exist_ok=True)

        route_start_time = time.time()
        self.logger.info(
            f"Processing route {route_id}: {start_location} to {end_location}")

        results = {
            "route_id": route_id,
            "start_location": start_location,
            "end_location": end_location,
            "success": False,
            "stages": {},
            "stats": {
                "start_time": route_start_time,
                "end_time": None,
                "duration": None,
                "total_images": 0,
                "total_detections": 0
            }
        }

        try:
            # 1. Plan route
            self.logger.info(f"[Route {route_id}] Planning route...")
            route_data = self.route_planner.plan_route(
                start_location, end_location)

            if not route_data.get("success", False):
                self.logger.error(
                    f"[Route {route_id}] Route planning failed: {route_data.get('error', 'Unknown error')}")
                results["error"] = f"Route planning failed: {route_data.get('error', 'Unknown error')}"
                return results

            results["stages"]["route_planning"] = {
                "success": True,
                "route_points": len(route_data["route_points"]),
                "bearings": len(route_data["bearings"])
            }

            # 2. Acquire street view images
            self.logger.info(f"[Route {route_id}] Acquiring images...")
            streetview_client = StreetViewClient(self.config)
            image_results = await streetview_client.get_route_images(route_data)

            if not image_results.get("success", False):
                self.logger.error(
                    f"[Route {route_id}] Image acquisition failed: {image_results.get('error', 'Unknown error')}")
                results["error"] = f"Image acquisition failed: {image_results.get('error', 'Unknown error')}"
                return results

            # Save the images if requested
            if save_images:
                images_dir = os.path.join(route_dir, "images")
                os.makedirs(images_dir, exist_ok=True)
                save_results = streetview_client.save_images(
                    image_results, images_dir)

            results["stages"]["image_acquisition"] = {
                "success": True,
                "total_images": image_results["metadata"]["total_images_requested"],
                "successful_images": image_results["metadata"]["successful_images"],
                "cached_images": image_results["metadata"]["cached_images"]
            }

            # 3. Process images
            self.logger.info(f"[Route {route_id}] Processing images...")
            image_processor = ImageProcessor(self.config)
            process_results = image_processor.batch_process_images(
                image_results["images"])

            if not process_results.get("success", False):
                self.logger.error(
                    f"[Route {route_id}] Image processing failed")
                results["error"] = "Image processing failed"
                return results

            results["stages"]["image_processing"] = {
                "success": True,
                "processed_images": len(process_results["processed_images"]),
                "augmented_images": len(process_results["augmented_images"]),
                "failed_images": len(process_results["failed_images"])
            }

            # 4. Detect sleeves
            self.logger.info(f"[Route {route_id}] Detecting sleeves...")
            detection_batch = self.detector.batch_detect(
                process_results["processed_images"], use_ensemble=True)

            if not detection_batch.get("success", False):
                self.logger.error(
                    f"[Route {route_id}] Sleeve detection failed")
                results["error"] = "Sleeve detection failed"
                return results

            # Save detection visualizations if requested
            if draw_detections:
                detections_dir = os.path.join(route_dir, "detections")
                os.makedirs(detections_dir, exist_ok=True)
                vis_results = self.detector.save_detection_results(
                    detection_batch, detections_dir)

            total_detections = detection_batch["metadata"]["total_detections"]

            results["stages"]["sleeve_detection"] = {
                "success": True,
                "total_detections": total_detections,
                "successful_images": detection_batch["metadata"]["successful_images"],
                "failed_images": detection_batch["metadata"]["failed_images"]
            }

            # 5. Track detections
            self.logger.info(f"[Route {route_id}] Tracking detections...")
            tracking_results = self.tracker.process_batch_results(
                detection_batch)

            results["stages"]["tracking"] = {
                "success": True,
                "unique_objects": tracking_results["metadata"]["unique_objects"],
                "geo_referenced_detections": tracking_results["metadata"]["geo_referenced_detections"]
            }

            # 6. Generate coverage map if requested
            if generate_map:
                self.logger.info(
                    f"[Route {route_id}] Generating coverage map...")
                # Convert route points to the format expected by MapGenerator
                map_route_points = [{"latitude": lat, "longitude": lng}
                                    for lat, lng in route_data["route_points"]]

                # Convert image points to format expected by MapGenerator
                image_points = []
                for img in image_results["images"]:
                    if img.get("success", False):
                        params = img["params"]
                        image_points.append({
                            "latitude": params.get("latitude", 0),
                            "longitude": params.get("longitude", 0),
                            "heading": params.get("heading", 0),
                            "fov": params.get("fov", 55),
                            "point_index": params.get("point_index", 0)
                        })

                # Prepare detection results for map
                map_detections = []
                for result in detection_batch["detection_results"]:
                    if result["detection_result"]["detections"]:
                        map_detections.append({
                            "image_params": result["image_params"],
                            "detection_result": {
                                "detections": result["detection_result"]["detections"]
                            }
                        })

                # Generate map
                maps_dir = os.path.join(route_dir, "maps")
                os.makedirs(maps_dir, exist_ok=True)
                self.map_generator.output_dir = maps_dir

                map_path = self.map_generator.generate_map(
                    map_route_points, image_points, map_detections)

                # Also generate heatmap if we have detections
                if total_detections > 0:
                    heatmap_path = self.map_generator.generate_detection_heatmap(
                        map_detections)
                    results["stages"]["visualization"] = {
                        "success": True,
                        "map_path": map_path,
                        "heatmap_path": heatmap_path
                    }
                else:
                    results["stages"]["visualization"] = {
                        "success": True,
                        "map_path": map_path
                    }

                # Export to GeoJSON for compatibility with other GIS tools
                geojson_path = self.map_generator.export_geojson(
                    map_route_points, map_detections)
                if geojson_path:
                    results["stages"]["visualization"]["geojson_path"] = geojson_path

            # 7. Save detailed results
            results_file = os.path.join(route_dir, "route_results.json")
            with open(results_file, "w") as f:
                # Create a more compact version of the results for storage
                compact_results = {
                    "route_id": route_id,
                    "start_location": start_location,
                    "end_location": end_location,
                    "success": True,
                    "stats": {
                        "route_points": len(route_data["route_points"]),
                        "total_images": image_results["metadata"]["total_images_requested"],
                        "successful_images": image_results["metadata"]["successful_images"],
                        "total_detections": total_detections,
                        "unique_objects": tracking_results["metadata"]["unique_objects"]
                    },
                    "stages": results["stages"]
                }
                json.dump(compact_results, f, indent=2)

            # 8. Export detection data to CSV for further analysis
            if total_detections > 0:
                csv_path = os.path.join(route_dir, "detections.csv")
                self.tracker.save_to_csv(csv_path)
                results["stages"]["export"] = {
                    "success": True,
                    "csv_path": csv_path
                }

            # Update final statistics
            route_end_time = time.time()
            results["success"] = True
            results["stats"]["end_time"] = route_end_time
            results["stats"]["duration"] = route_end_time - route_start_time
            results["stats"]["total_images"] = image_results["metadata"]["successful_images"]
            results["stats"]["total_detections"] = total_detections

            # Store route results for summary
            self.detection_results[route_id] = results
            self.routes_processed += 1

            self.logger.info(f"[Route {route_id}] Processing completed in {results['stats']['duration']:.2f}s. " +
                             f"Found {total_detections} detections in {results['stats']['total_images']} images.")

            return results

        except Exception as e:
            self.logger.error(
                f"[Route {route_id}] Processing failed with error: {e}", exc_info=True)
            results["error"] = str(e)
            return results

    async def process_routes_sequential(self, routes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process multiple routes sequentially.

        Args:
            routes: List of route dictionaries with route_id, start_location, end_location

        Returns:
            Dictionary with batch processing results
        """
        batch_start_time = time.time()
        self.logger.info(f"Processing {len(routes)} routes sequentially")

        results = []
        for route in routes:
            route_id = route["route_id"]
            start_location = route["start_location"]
            end_location = route["end_location"]

            route_result = await self.process_route(
                route_id, start_location, end_location,
                save_images=route.get("save_images", True),
                draw_detections=route.get("draw_detections", True),
                generate_map=route.get("generate_map", True)
            )

            results.append(route_result)

        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time

        summary = self._generate_batch_summary(results, batch_duration)

        return {
            "success": True,
            "route_results": results,
            "summary": summary,
            "duration": batch_duration
        }

    async def process_routes_parallel(self, routes: List[Dict[str, Any]], max_concurrent: int = 2) -> Dict[str, Any]:
        """
        Process multiple routes in parallel with a limit on concurrent operations.

        Args:
            routes: List of route dictionaries with route_id, start_location, end_location
            max_concurrent: Maximum number of routes to process concurrently

        Returns:
            Dictionary with batch processing results
        """
        batch_start_time = time.time()
        self.logger.info(
            f"Processing {len(routes)} routes in parallel (max_concurrent={max_concurrent})")

        # Process routes in batches to limit concurrency
        all_results = []
        for i in range(0, len(routes), max_concurrent):
            batch = routes[i:i+max_concurrent]
            self.logger.info(f"Processing batch of {len(batch)} routes")

            # Create tasks for each route in this batch
            tasks = []
            for route in batch:
                route_id = route["route_id"]
                start_location = route["start_location"]
                end_location = route["end_location"]

                task = self.process_route(
                    route_id, start_location, end_location,
                    save_images=route.get("save_images", True),
                    draw_detections=route.get("draw_detections", True),
                    generate_map=route.get("generate_map", True)
                )
                tasks.append(task)

            # Run all tasks in this batch concurrently
            batch_results = await asyncio.gather(*tasks)
            all_results.extend(batch_results)

            # Give system time to clean up resources
            await asyncio.sleep(1)

        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time

        summary = self._generate_batch_summary(all_results, batch_duration)

        return {
            "success": True,
            "route_results": all_results,
            "summary": summary,
            "duration": batch_duration
        }

    def _generate_batch_summary(self, results: List[Dict[str, Any]], duration: float) -> Dict[str, Any]:
        """
        Generate summary statistics for a batch of processed routes.

        Args:
            results: List of route processing results
            duration: Total processing duration

        Returns:
            Dictionary with summary statistics
        """
        total_routes = len(results)
        successful_routes = sum(1 for r in results if r.get("success", False))
        failed_routes = total_routes - successful_routes

        total_images = sum(r.get("stats", {}).get("total_images", 0)
                           for r in results if r.get("success", False))
        total_detections = sum(r.get("stats", {}).get(
            "total_detections", 0) for r in results if r.get("success", False))

        # Calculate cache statistics
        cache_stats = self.cache_manager.get_stats()

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_routes": total_routes,
            "successful_routes": successful_routes,
            "failed_routes": failed_routes,
            "total_images": total_images,
            "total_detections": total_detections,
            "duration_seconds": duration,
            "average_detection_per_route": total_detections / successful_routes if successful_routes > 0 else 0,
            "average_images_per_route": total_images / successful_routes if successful_routes > 0 else 0,
            "cache_stats": cache_stats
        }

        # Save summary to file
        summary_file = os.path.join(self.output_dir, "batch_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Also save a CSV with route results for easy analysis
        csv_file = os.path.join(self.output_dir, "route_results.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Route ID", "Success", "Start Location", "End Location",
                            "Total Images", "Total Detections", "Duration (s)"])

            for r in results:
                writer.writerow([
                    r.get("route_id", "unknown"),
                    r.get("success", False),
                    r.get("start_location", ""),
                    r.get("end_location", ""),
                    r.get("stats", {}).get("total_images", 0),
                    r.get("stats", {}).get("total_detections", 0),
                    r.get("stats", {}).get("duration", 0)
                ])

        self.logger.info(f"Batch processing completed: {successful_routes}/{total_routes} routes successful, " +
                         f"{total_detections} total detections in {total_images} images. " +
                         f"Duration: {duration:.2f}s")

        return summary

    def load_routes_from_csv(self, csv_file: str) -> List[Dict[str, Any]]:
        """
        Load route definitions from a CSV file.

        Args:
            csv_file: Path to CSV file with route definitions

        Returns:
            List of route dictionaries
        """
        if not os.path.exists(csv_file):
            self.logger.error(f"Routes CSV file not found: {csv_file}")
            return []

        try:
            routes = []
            df = pd.read_csv(csv_file)

            # Check required columns
            required_columns = ["route_id", "start_location", "end_location"]
            missing_columns = [
                col for col in required_columns if col not in df.columns]

            if missing_columns:
                self.logger.error(
                    f"Missing required columns in CSV: {missing_columns}")
                return []

            # Convert to list of dictionaries
            for _, row in df.iterrows():
                route = {
                    "route_id": str(row["route_id"]),
                    "start_location": row["start_location"],
                    "end_location": row["end_location"]
                }

                # Add optional columns if present
                for col in ["save_images", "draw_detections", "generate_map"]:
                    if col in df.columns:
                        route[col] = bool(row[col])

                routes.append(route)

            self.logger.info(f"Loaded {len(routes)} routes from {csv_file}")
            return routes

        except Exception as e:
            self.logger.error(f"Error loading routes from CSV: {e}")
            return []

    def create_route_subset_for_validation(self, routes: List[Dict[str, Any]],
                                           num_routes: int = 5,
                                           output_csv: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Create a subset of routes for validation before running on all routes.

        Args:
            routes: List of route dictionaries
            num_routes: Number of routes to include in the subset
            output_csv: Optional path to save the subset to CSV

        Returns:
            List of route dictionaries for the subset
        """
        if not routes:
            return []

        import random

        # If fewer routes than requested, use all of them
        if len(routes) <= num_routes:
            subset = routes
        else:
            # Randomly select routes for the subset
            subset = random.sample(routes, num_routes)

        # Save to CSV if requested
        if output_csv:
            df = pd.DataFrame(subset)
            df.to_csv(output_csv, index=False)
            self.logger.info(
                f"Saved validation subset with {len(subset)} routes to {output_csv}")

        return subset

    def export_combined_results(self, output_file: Optional[str] = None) -> str:
        """
        Export combined detection results from all routes to a single GeoJSON file.

        Args:
            output_file: Optional path for the output file

        Returns:
            Path to the output file
        """
        if not self.detection_results:
            self.logger.warning(
                "No routes processed yet, cannot export results")
            return None

        try:
            import geopandas as gpd
            from shapely.geometry import Point

            # Get all detection points across all routes
            all_detections = []

            for route_id, result in self.detection_results.items():
                if not result.get("success", False):
                    continue

                # Get route path
                route_dir = os.path.join(self.output_dir, f"route_{route_id}")
                csv_path = os.path.join(route_dir, "detections.csv")

                if os.path.exists(csv_path):
                    # Load detections from this route
                    df = pd.read_csv(csv_path)
                    df["route_id"] = route_id
                    all_detections.append(df)

            if not all_detections:
                self.logger.warning("No detection results found across routes")
                return None

            # Combine all dataframes
            combined_df = pd.concat(all_detections, ignore_index=True)

            # Create GeoDataFrame
            geometry = [Point(row.image_longitude, row.image_latitude)
                        for _, row in combined_df.iterrows()]

            gdf = gpd.GeoDataFrame(
                combined_df, geometry=geometry, crs="EPSG:4326")

            # Save to GeoJSON
            if output_file is None:
                output_file = os.path.join(
                    self.output_dir, "combined_detections.geojson")

            gdf.to_file(output_file, driver="GeoJSON")
            self.logger.info(
                f"Exported {len(gdf)} combined detections to {output_file}")

            return output_file

        except ImportError:
            self.logger.error(
                "Cannot export combined results: geopandas not installed")
            return None
        except Exception as e:
            self.logger.error(f"Error exporting combined results: {e}")
            return None
