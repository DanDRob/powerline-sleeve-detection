import asyncio
import json
import aiohttp
import time
from io import BytesIO
from typing import Dict, List, Tuple, Any, Optional, Union
from PIL import Image
import numpy as np
import os

from ..system.config import Config
from ..system.logging import get_logger
from .cache_manager import CacheManager


class StreetViewClient:
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("streetview_client")
        self.session = None
        self.last_request_time = 0
        self.request_lock = asyncio.Lock()

        # Initialize cache manager
        self.cache_manager = CacheManager(config)
        self.logger.info(f"Cache enabled: {self.cache_manager.enabled}")

    async def initialize(self):
        """Initialize HTTP session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            self.logger.info("Initialized HTTP session")

    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info("Closed HTTP session")

    async def _apply_rate_limiting(self):
        """Apply rate limiting to avoid API throttling"""
        async with self.request_lock:
            current_time = time.time()
            elapsed = current_time - self.last_request_time

            if elapsed < self.config.api.min_request_interval:
                delay = self.config.api.min_request_interval - elapsed
                self.logger.debug(f"Rate limiting: waiting {delay:.2f}s")
                await asyncio.sleep(delay)

            self.last_request_time = time.time()

    def _get_params_for_index(self, index: int, headings: List[float],
                              pitches: List[float], fovs: List[float]) -> Tuple[float, float, float]:
        """Calculate which heading, pitch, fov combination corresponds to a result index"""
        pitch_fov_count = len(pitches) * len(fovs)
        heading_index = index // pitch_fov_count
        remainder = index % pitch_fov_count
        pitch_index = remainder // len(fovs)
        fov_index = remainder % len(fovs)

        return headings[heading_index], pitches[pitch_index], fovs[fov_index]

    async def get_streetview_image(self, lat: float, lng: float, heading: float,
                                   pitch: float, fov: float) -> Dict[str, Any]:
        """
        Fetch a single street view image asynchronously.

        Args:
            lat: Latitude
            lng: Longitude
            heading: Camera heading in degrees
            pitch: Camera pitch in degrees
            fov: Field of view in degrees

        Returns:
            Dictionary with image data and parameters
        """
        # Create cache key
        cache_key = {
            "type": "streetview_image",
            "lat": lat,
            "lng": lng,
            "heading": heading,
            "pitch": pitch,
            "fov": fov
        }

        # Check cache first
        cached_image_data = self.cache_manager.get(cache_key, "images")
        if cached_image_data is not None:
            self.logger.debug(
                f"Cache hit for StreetView image at {lat},{lng}, heading={heading}")
            return {
                "image": cached_image_data,
                "params": {
                    "latitude": lat,
                    "longitude": lng,
                    "heading": heading,
                    "pitch": pitch,
                    "fov": fov
                },
                "success": True,
                "cached": True
            }

        # Ensure session is initialized
        if self.session is None:
            await self.initialize()

        base_url = "https://maps.googleapis.com/maps/api/streetview"
        params = {
            "size": f"{self.config.api.image_width}x{self.config.api.image_height}",
            "location": f"{lat},{lng}",
            "heading": heading,
            "pitch": pitch,
            "fov": fov,
            "key": self.config.api.key,
        }

        # Apply rate limiting
        await self._apply_rate_limiting()

        # Handle retries
        for attempt in range(self.config.api.max_retries):
            try:
                async with self.session.get(base_url, params=params) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        try:
                            image = Image.open(BytesIO(image_data))

                            # Cache the image
                            self.cache_manager.set(cache_key, image, "images")

                            return {
                                "image": image,
                                "params": {
                                    "latitude": lat,
                                    "longitude": lng,
                                    "heading": heading,
                                    "pitch": pitch,
                                    "fov": fov
                                },
                                "success": True,
                                "cached": False
                            }
                        except Exception as e:
                            self.logger.error(
                                f"Failed to process image data: {e}")
                            return {
                                "image": None,
                                "params": {
                                    "latitude": lat,
                                    "longitude": lng,
                                    "heading": heading,
                                    "pitch": pitch,
                                    "fov": fov
                                },
                                "success": False,
                                "error": f"Image processing error: {e}"
                            }
                    else:
                        error_text = await response.text()
                        self.logger.warning(
                            f"HTTP {response.status} for {lat},{lng}: {error_text}")

                        if attempt < self.config.api.max_retries - 1:
                            delay = self.config.api.retry_delay * \
                                (2 ** attempt)
                            self.logger.debug(
                                f"Retrying after {delay}s (attempt {attempt+1}/{self.config.api.max_retries})")
                            await asyncio.sleep(delay)
                        else:
                            return {
                                "image": None,
                                "params": {
                                    "latitude": lat,
                                    "longitude": lng,
                                    "heading": heading,
                                    "pitch": pitch,
                                    "fov": fov
                                },
                                "success": False,
                                "error": f"HTTP {response.status}: {error_text}"
                            }

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                self.logger.warning(f"Request failed for {lat},{lng}: {e}")

                if attempt < self.config.api.max_retries - 1:
                    delay = self.config.api.retry_delay * (2 ** attempt)
                    self.logger.debug(
                        f"Retrying after {delay}s (attempt {attempt+1}/{self.config.api.max_retries})")
                    await asyncio.sleep(delay)
                else:
                    return {
                        "image": None,
                        "params": {
                            "latitude": lat,
                            "longitude": lng,
                            "heading": heading,
                            "pitch": pitch,
                            "fov": fov
                        },
                        "success": False,
                        "error": f"Request failed: {e}"
                    }

    async def get_streetview_images(self, location: Tuple[float, float],
                                    headings: List[float], pitches: List[float],
                                    fovs: List[float]) -> List[Dict[str, Any]]:
        """
        Asynchronously fetch multiple street view images.

        Args:
            location: (latitude, longitude) tuple
            headings: List of camera headings in degrees
            pitches: List of camera pitches in degrees
            fovs: List of fields of view in degrees

        Returns:
            List of dictionaries with image data and parameters
        """
        # Ensure session is initialized
        if self.session is None:
            await self.initialize()

        tasks = []
        lat, lng = location

        self.logger.info(
            f"Fetching {len(headings) * len(pitches) * len(fovs)} images at {lat},{lng}")

        for heading in headings:
            for pitch in pitches:
                for fov in fovs:
                    task = self.get_streetview_image(
                        lat, lng, heading, pitch, fov)
                    tasks.append(task)

        # Execute all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        valid_results = []
        cached_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                heading, pitch, fov = self._get_params_for_index(
                    i, headings, pitches, fovs)
                self.logger.error(
                    f"Exception for {lat},{lng} with heading {heading}, pitch {pitch}, fov {fov}: {result}")
                valid_results.append({
                    "image": None,
                    "params": {
                        "latitude": lat,
                        "longitude": lng,
                        "heading": heading,
                        "pitch": pitch,
                        "fov": fov
                    },
                    "success": False,
                    "error": f"Exception: {result}"
                })
            else:
                valid_results.append(result)
                if result.get("cached", False):
                    cached_count += 1

        successful_count = sum(1 for r in valid_results if r['success'])
        self.logger.info(
            f"Fetched {successful_count} successful images ({cached_count} from cache) out of {len(valid_results)}")
        return valid_results

    async def get_route_images(self, route_data: Dict[str, Any],
                               right_side: bool = True,
                               left_side: bool = True) -> Dict[str, Any]:
        """
        Fetch all street view images along a route.

        Args:
            route_data: Route data from RoutePlanner
            right_side: Whether to capture right-side images
            left_side: Whether to capture left-side images

        Returns:
            Dictionary with all image data and metadata
        """
        if not route_data.get("success", False):
            self.logger.error(
                "Cannot process route data; route planning failed")
            return {"success": False, "error": "Route planning failed"}

        # Extract route information
        route_points = route_data["route_points"]
        bearings = route_data["bearings"]

        # Prepare storage for results
        results = {
            "images": [],
            "metadata": {
                "total_points": len(route_points),
                "total_images_requested": 0,
                "successful_images": 0,
                "failed_images": 0,
                "cached_images": 0
            },
            "success": True
        }

        # Configure camera angles
        camera_config = self.config.camera

        # Process each point on the route
        for i, (lat, lng) in enumerate(route_points):
            self.logger.info(
                f"Processing point {i+1}/{len(route_points)}: ({lat}, {lng})")

            # Get the road heading at this point
            road_heading = bearings[i] if i < len(bearings) else bearings[-1]
            self.logger.debug(f"Road heading: {road_heading}")

            # Define camera parameters
            all_images = []

            # Right side images
            if right_side:
                for angle in camera_config.relative_angles_right:
                    # Use special FOV for diagonal angles
                    fov_values = camera_config.fov_special if angle in [
                        45, 135] else camera_config.fov_default
                    transmission_heading = (road_heading + angle) % 360

                    # Fetch images for this angle
                    images = await self.get_streetview_images(
                        (lat, lng),
                        [transmission_heading],
                        camera_config.pitch_values,
                        fov_values
                    )
                    all_images.extend(images)

            # Left side images
            if left_side:
                for angle in camera_config.relative_angles_left:
                    # Use special FOV for diagonal angles
                    fov_values = camera_config.fov_special if angle in [
                        310, 230] else camera_config.fov_default
                    transmission_heading = (road_heading + angle) % 360

                    # Fetch images for this angle
                    images = await self.get_streetview_images(
                        (lat, lng),
                        [transmission_heading],
                        camera_config.pitch_values,
                        fov_values
                    )
                    all_images.extend(images)

            # Update results
            for result in all_images:
                if result["success"]:
                    results["metadata"]["successful_images"] += 1
                    if result.get("cached", False):
                        results["metadata"]["cached_images"] += 1
                else:
                    results["metadata"]["failed_images"] += 1

                # Add point index and road heading to parameters
                result["params"]["point_index"] = i
                result["params"]["road_heading"] = road_heading

                # Include everything in the results
                results["images"].append(result)

            results["metadata"]["total_images_requested"] += len(all_images)

        # Calculate cache statistics
        cache_stats = self.cache_manager.get_stats()
        results["metadata"]["cache_stats"] = cache_stats

        cache_percentage = 0
        if results["metadata"]["successful_images"] > 0:
            cache_percentage = (
                results["metadata"]["cached_images"] / results["metadata"]["successful_images"]) * 100

        self.logger.info(f"Completed image acquisition. "
                         f"Success rate: {results['metadata']['successful_images']}/{results['metadata']['total_images_requested']} "
                         f"({results['metadata']['successful_images']/results['metadata']['total_images_requested']*100:.1f}%) "
                         f"Cache hits: {results['metadata']['cached_images']} ({cache_percentage:.1f}%)")

        return results

    def save_images(self, image_results: Dict[str, Any], output_dir: str = None) -> Dict[str, Any]:
        """
        Save acquired images to disk.

        Args:
            image_results: Results from get_route_images
            output_dir: Directory to save images (uses config if None)

        Returns:
            Dictionary with file paths and metadata
        """
        if output_dir is None:
            output_dir = os.path.join(self.config.system.output_dir, "images")

        os.makedirs(output_dir, exist_ok=True)

        saved_files = []
        for i, result in enumerate(image_results.get("images", [])):
            if not result.get("success", False) or result.get("image") is None:
                continue

            # Extract parameters
            params = result["params"]
            point_index = params.get("point_index", i)
            heading = params.get("heading", 0)
            pitch = params.get("pitch", 0)
            fov = params.get("fov", 55)

            # Determine side (right or left)
            road_heading = params.get("road_heading", 0)
            relative_heading = (heading - road_heading) % 360
            side = "right" if relative_heading in [45, 90, 135] else "left"

            # Create filename
            filename = f"snapshot_{side}_{point_index}_heading_{int(heading)}_pitch_{int(pitch)}_fov_{int(fov)}.jpg"
            filepath = os.path.join(output_dir, filename)

            # Save image
            try:
                result["image"].save(filepath)
                saved_files.append({
                    "filepath": filepath,
                    "filename": filename,
                    "params": params,
                    "side": side,
                    "cached": result.get("cached", False)
                })
                self.logger.debug(f"Saved {filename}")
            except Exception as e:
                self.logger.error(f"Failed to save {filename}: {e}")

        # Add cache info to saved summary
        cached_saves = sum(1 for f in saved_files if f.get("cached", False))
        self.logger.info(
            f"Saved {len(saved_files)} images to {output_dir} ({cached_saves} were from cache)")

        return {
            "success": True,
            "saved_files": saved_files,
            "metadata": {
                "total_saved": len(saved_files),
                "cached_saved": cached_saves,
                "output_dir": output_dir
            }
        }
