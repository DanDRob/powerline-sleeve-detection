import requests
import polyline
import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import geopy.distance
from pyproj import Geod
import time
from ..system.config import Config
from ..system.logging import get_logger


class RoutePlanner:
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("route_planner")
        self.geod = Geod(ellps='WGS84')

    def get_route_coordinates(self, start_location: str, end_location: str) -> List[Tuple[float, float]]:
        """
        Retrieve route coordinates from Google Directions API.

        Args:
            start_location: Starting location (address or lat,lng)
            end_location: Ending location (address or lat,lng)

        Returns:
            List of (latitude, longitude) tuples representing the route
        """
        base_url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            'origin': start_location,
            'destination': end_location,
            'key': self.config.api.key,
        }

        self.logger.info(
            f"Fetching route from {start_location} to {end_location}")

        for attempt in range(self.config.api.max_retries):
            try:
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if not data.get('routes'):
                    self.logger.error(
                        "No routes found in the Directions API response")
                    return []

                coordinates = []
                for leg in data['routes'][0]['legs']:
                    for step in leg['steps']:
                        points = step['polyline']['points']
                        decoded_points = polyline.decode(points)
                        coordinates.extend(decoded_points)

                self.logger.info(
                    f"Successfully retrieved route with {len(coordinates)} points")
                return coordinates

            except requests.exceptions.RequestException as e:
                self.logger.warning(
                    f"API request failed (attempt {attempt+1}/{self.config.api.max_retries}): {e}")
                if attempt < self.config.api.max_retries - 1:
                    time.sleep(self.config.api.retry_delay *
                               (2 ** attempt))  # Exponential backoff
                else:
                    self.logger.error(
                        "Failed to get route after multiple attempts")
                    return []

    def calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the bearing between two points in degrees."""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * \
            math.cos(lat2) * math.cos(dlon)
        initial_bearing = math.atan2(y, x)
        compass_bearing = (math.degrees(initial_bearing) + 360) % 360
        return compass_bearing

    def haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate Haversine distance between two points in kilometers."""
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
        a = math.sin(delta_lat / 2)**2 + math.cos(lat1) * \
            math.cos(lat2) * math.sin(delta_lon / 2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Earth's radius in kilometers
        return r * c

    def interpolate_route(self, points: List[Tuple[float, float]], interval: float = None) -> List[Tuple[float, float]]:
        """
        Interpolate points along a route at a specified interval in meters.

        Args:
            points: List of (latitude, longitude) tuples
            interval: Distance between points in meters (uses config if None)

        Returns:
            List of interpolated (latitude, longitude) tuples
        """
        if interval is None:
            interval = self.config.sampling.base_interval

        if not points:
            self.logger.warning("No points provided for interpolation")
            return []

        self.logger.info(
            f"Interpolating route with {len(points)} points at {interval}m intervals")

        new_points = [points[0]]
        for i in range(len(points) - 1):
            lat1, lon1 = points[i]
            lat2, lon2 = points[i + 1]
            distance = self.haversine(
                lat1, lon1, lat2, lon2) * 1000  # Convert to meters

            if distance < interval:
                continue

            bearing = self.calculate_bearing(lat1, lon1, lat2, lon2)
            num_steps = int(distance // interval)

            for step in range(1, num_steps + 1):
                new_point = geopy.distance.distance(
                    meters=interval * step).destination((lat1, lon1), bearing)
                new_points.append((new_point.latitude, new_point.longitude))

        # Add the last point if it's not already included
        if new_points[-1] != points[-1]:
            new_points.append(points[-1])

        self.logger.info(f"Interpolated route with {len(new_points)} points")
        return new_points

    def calculate_segment_importance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """
        Calculate importance factor for a segment based on multiple criteria.

        Args:
            p1: First point (latitude, longitude)
            p2: Second point (latitude, longitude)

        Returns:
            Importance factor between 0 and 1
        """

        # Check if there's a significant bearing change
        lat1, lon1 = p1
        lat2, lon2 = p2

        # If we have at least 3 points in our context, calculate bearing change
        bearing_change_factor = 0.0
        distance_factor = 0.0

        # Calculate distance factor (longer segments get higher importance)
        distance = self.haversine(lat1, lon1, lat2, lon2) * 1000  # meters
        if distance > 0:
            # Normalize to 0-1 range, with distances over 100m getting full weight
            distance_factor = min(distance / 100.0, 1.0)

        # Combine factors with weights from config
        importance = (
            self.config.sampling.weights.infrastructure * distance_factor +
            self.config.sampling.weights.environmental * 0.5 +
            self.config.sampling.weights.historical * 0.5
        )

        return min(max(importance, 0.0), 1.0)  # Ensure between 0 and 1

    def apply_adaptive_sampling(self, base_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Apply adaptive sampling to add more points in important areas.

        Args:
            base_points: Base interpolated points

        Returns:
            Enhanced points with adaptive sampling
        """
        if len(base_points) < 2:
            return base_points

        self.logger.info(
            f"Applying adaptive sampling to {len(base_points)} points")

        enhanced_points = []

        for i in range(len(base_points) - 1):
            p1 = base_points[i]
            p2 = base_points[i + 1]

            # Add the first point
            enhanced_points.append(p1)

            # Check if this segment needs dense sampling
            importance = self.calculate_segment_importance(p1, p2)

            if importance > self.config.sampling.importance_threshold:
                # Add additional points in this segment
                num_extra_points = int(
                    importance * self.config.sampling.max_extra_points)
                self.logger.debug(
                    f"Adding {num_extra_points} extra points between {p1} and {p2} (importance: {importance:.2f})")

                for j in range(1, num_extra_points + 1):
                    ratio = j / (num_extra_points + 1)
                    lat = p1[0] + (p2[0] - p1[0]) * ratio
                    lon = p1[1] + (p2[1] - p1[1]) * ratio
                    enhanced_points.append((lat, lon))

        # Add the last point
        enhanced_points.append(base_points[-1])

        self.logger.info(
            f"Adaptive sampling added {len(enhanced_points) - len(base_points)} points")
        return enhanced_points

    def calculate_route_bearings(self, points: List[Tuple[float, float]]) -> List[float]:
        """
        Calculate bearings for each segment of the route.

        Args:
            points: List of (latitude, longitude) tuples

        Returns:
            List of bearings in degrees
        """
        if len(points) < 2:
            return []

        bearings = []
        for i in range(len(points) - 1):
            lat1, lon1 = points[i]
            lat2, lon2 = points[i + 1]
            bearing = self.calculate_bearing(lat1, lon1, lat2, lon2)
            bearings.append(bearing)

        # For the last point, use the last bearing
        bearings.append(bearings[-1] if bearings else 0)

        return bearings

    def calculate_powerline_points(
        self,
        route_points: List[Tuple[float, float]],
        bearings: List[float],
        offset_distance: float = None,
        side: str = None
    ) -> List[Tuple[float, float]]:
        """
        Calculate powerline positions with precise geodetic offsets.

        Args:
            route_points: List of (latitude, longitude) tuples for the route
            bearings: List of bearings for each point
            offset_distance: Distance in meters to offset (uses config if None)
            side: Side to offset ("right", "left", or "both"; uses config if None)

        Returns:
            List of (latitude, longitude) tuples for powerline positions
        """
        if offset_distance is None:
            offset_distance = self.config.powerline.offset_distance

        if side is None:
            side = self.config.powerline.side

        if len(route_points) != len(bearings):
            self.logger.error(
                f"Mismatch in route points ({len(route_points)}) and bearings ({len(bearings)})")
            # Adjust bearings if possible
            if len(route_points) > len(bearings) and len(bearings) > 0:
                bearings.extend([bearings[-1]] *
                                (len(route_points) - len(bearings)))
            else:
                return []

        powerline_points = []
        sides_to_process = []

        if side == "both":
            sides_to_process = ["right", "left"]
        else:
            sides_to_process = [side]

        for current_side in sides_to_process:
            side_points = []
            for i, (lat, lon) in enumerate(route_points):
                bearing = bearings[i]
                perp_bearing = (
                    bearing + 90) % 360 if current_side == "right" else (bearing + 270) % 360

                # Calculate offset point using pyproj Geod
                lon_pl, lat_pl, _ = self.geod.fwd(
                    lon, lat, perp_bearing, offset_distance)
                side_points.append((lat_pl, lon_pl))

            powerline_points.extend(side_points)

        self.logger.info(f"Generated {len(powerline_points)} powerline points")
        return powerline_points

    def plan_route(self, start_location: str, end_location: str) -> Dict[str, Any]:
        """
        Plan a route with all necessary points and metadata.

        Args:
            start_location: Starting location (address or lat,lng)
            end_location: Ending location (address or lat,lng)

        Returns:
            Dictionary containing route data
        """
        self.logger.info(
            f"Planning route from {start_location} to {end_location}")

        # Get basic route from Google
        route_points = self.get_route_coordinates(start_location, end_location)
        if not route_points:
            self.logger.error("Failed to get route coordinates")
            return {"success": False, "error": "Failed to get route coordinates"}

        # Interpolate route points at base interval
        interpolated_points = self.interpolate_route(route_points)
        if not interpolated_points:
            self.logger.error("Failed to interpolate route points")
            return {"success": False, "error": "Failed to interpolate route points"}

        # Apply adaptive sampling
        enhanced_points = self.apply_adaptive_sampling(interpolated_points)

        # Calculate bearings
        bearings = self.calculate_route_bearings(enhanced_points)

        # Calculate powerline points
        powerline_points = self.calculate_powerline_points(
            enhanced_points, bearings)

        # Interpolate powerline points at a finer interval for coverage analysis
        interpolated_powerline_points = self.interpolate_route(
            powerline_points, interval=1.0)

        result = {
            "success": True,
            "route_points": enhanced_points,
            "bearings": bearings,
            "powerline_points": powerline_points,
            "interpolated_powerline_points": interpolated_powerline_points,
            "start": start_location,
            "end": end_location,
            "metadata": {
                "total_route_points": len(enhanced_points),
                "total_powerline_points": len(powerline_points),
                "total_interpolated_powerline_points": len(interpolated_powerline_points)
            }
        }

        self.logger.info(
            f"Route planning completed successfully with {len(enhanced_points)} points")
        return result

    def plan_routes_from_csv(self, csv_file: str) -> Dict[str, Any]:
        """
        Plan multiple routes from a CSV file.
        
        Args:
            csv_file: Path to CSV file containing route information
            
        Returns:
            Dictionary with planning results
        """
        import csv
        import os
        import json
        
        self.logger.info(f"Planning routes from CSV file: {csv_file}")
        
        if not os.path.exists(csv_file):
            self.logger.error(f"CSV file not found: {csv_file}")
            return {"success": False, "error": f"CSV file not found: {csv_file}"}
            
        results = []
        routes_processed = 0
        routes_succeeded = 0
        
        # Create output directory
        output_dir = os.path.join(self.config.system.output_dir, "route_plans")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            with open(csv_file, 'r') as f:
                csv_reader = csv.DictReader(f)
                
                for row in csv_reader:
                    route_id = row.get('route_id')
                    start_location = row.get('start_location')
                    end_location = row.get('end_location')
                    
                    if not all([route_id, start_location, end_location]):
                        self.logger.warning(f"Skipping row with missing data: {row}")
                        continue
                        
                    self.logger.info(f"Planning route {route_id}: {start_location} to {end_location}")
                    routes_processed += 1
                    
                    try:
                        # Plan the route
                        route_result = self.plan_route(start_location, end_location)
                        
                        if route_result.get('success', False):
                            routes_succeeded += 1
                            
                            # Save route plan to JSON file
                            output_file = os.path.join(output_dir, f"route_{route_id}.json")
                            with open(output_file, 'w') as out_f:
                                json.dump(route_result, out_f, indent=2)
                                
                            self.logger.info(f"Route {route_id} plan saved to {output_file}")
                            
                            # Add to results with file info
                            route_result['output_file'] = output_file
                            route_result['route_id'] = route_id
                            results.append(route_result)
                        else:
                            self.logger.error(f"Failed to plan route {route_id}: {route_result.get('error', 'Unknown error')}")
                            results.append({
                                "success": False,
                                "route_id": route_id,
                                "error": route_result.get('error', 'Unknown error')
                            })
                    except Exception as e:
                        self.logger.error(f"Error planning route {route_id}: {e}")
                        results.append({
                            "success": False,
                            "route_id": route_id,
                            "error": str(e)
                        })
            
            # Create summary file
            summary_file = os.path.join(output_dir, "routes_summary.json")
            summary = {
                "total_routes": routes_processed,
                "successful_routes": routes_succeeded,
                "failed_routes": routes_processed - routes_succeeded
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
                
            return {
                "success": True,
                "routes_processed": routes_processed,
                "routes_succeeded": routes_succeeded,
                "output_dir": output_dir,
                "summary_file": summary_file,
                "results": results
            }
                
        except Exception as e:
            self.logger.error(f"Error processing routes from CSV: {e}")
            return {"success": False, "error": str(e)}
