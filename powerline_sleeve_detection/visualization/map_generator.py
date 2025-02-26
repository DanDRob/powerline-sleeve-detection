import os
import json
import math
import folium
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import base64
import io
import time
import contextily as ctx
from shapely.geometry import Point, LineString
import geopandas as gpd

from ..system.config import Config
from ..system.logging import get_logger


class MapGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("map_generator")
        self.output_dir = os.path.join(config.system.output_dir, "maps")
        os.makedirs(self.output_dir, exist_ok=True)

        # Default center and zoom
        self.default_center = [43.6532, -79.3832]  # Toronto
        self.default_zoom = 13

        # Colors for different elements
        self.colors = {
            "route": "#3388FF",  # Blue for route line
            "coverage": "#00AA00",  # Green for covered segments
            "no_coverage": "#FF0000",  # Red for uncovered segments
            "image_points": "#3388FF",  # Blue for image acquisition points
            "detections": "#FF00FF",  # Magenta for sleeve detections
            "clusters": "#AA00AA",  # Purple for detection clusters
        }

        # Coverage status
        self.coverage = {
            "total_distance": 0,
            "covered_distance": 0,
            "coverage_percentage": 0
        }

    def create_base_map(self, route_points: List[Dict[str, float]] = None) -> folium.Map:
        """
        Create a base map centered on the route or default location.

        Args:
            route_points: Optional list of route points with lat/lng

        Returns:
            Folium map object
        """
        # Calculate center and bounds if route points provided
        if route_points and len(route_points) > 0:
            lats = [p.get("latitude", 0) for p in route_points]
            lngs = [p.get("longitude", 0) for p in route_points]

            # Filter out zero coordinates
            valid_lats = [lat for lat in lats if lat != 0]
            valid_lngs = [lng for lng in lngs if lng != 0]

            if valid_lats and valid_lngs:
                center = [sum(valid_lats) / len(valid_lats),
                          sum(valid_lngs) / len(valid_lngs)]
                bounds = [
                    [min(valid_lats), min(valid_lngs)],
                    [max(valid_lats), max(valid_lngs)]
                ]
            else:
                center = self.default_center
                bounds = None
        else:
            center = self.default_center
            bounds = None

        # Create base map
        map_obj = folium.Map(
            location=center,
            zoom_start=self.default_zoom,
            tiles="OpenStreetMap"
        )

        # Add alternative tile layers
        folium.TileLayer("Stamen Terrain").add_to(map_obj)
        folium.TileLayer("CartoDB Positron").add_to(map_obj)
        folium.TileLayer("CartoDB Dark_Matter").add_to(map_obj)

        # Fit bounds if available
        if bounds:
            map_obj.fit_bounds(bounds)

        # Add layer control
        folium.LayerControl().add_to(map_obj)

        return map_obj

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate haversine distance between two points.

        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates

        Returns:
            Distance in kilometers
        """
        R = 6371.0  # Earth radius in kilometers

        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Differences
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # Haversine formula
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * \
            math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c
        return distance

    def add_route_line(self, map_obj: folium.Map, route_points: List[Dict[str, float]]) -> folium.Map:
        """
        Add route line to the map.

        Args:
            map_obj: Folium map object
            route_points: List of route points with lat/lng

        Returns:
            Updated map object
        """
        if not route_points:
            return map_obj

        # Extract coordinates
        coordinates = []
        for point in route_points:
            lat = point.get("latitude")
            lng = point.get("longitude")
            if lat is not None and lng is not None and lat != 0 and lng != 0:
                coordinates.append([lat, lng])

        if not coordinates:
            self.logger.warning("No valid coordinates for route line")
            return map_obj

        # Create route line
        folium.PolyLine(
            locations=coordinates,
            color=self.colors["route"],
            weight=4,
            opacity=0.7,
            tooltip="Route"
        ).add_to(map_obj)

        # Add start and end markers
        if len(coordinates) > 0:
            folium.Marker(
                location=coordinates[0],
                icon=folium.Icon(color="green", icon="play"),
                tooltip="Start"
            ).add_to(map_obj)

            folium.Marker(
                location=coordinates[-1],
                icon=folium.Icon(color="red", icon="stop"),
                tooltip="End"
            ).add_to(map_obj)

        # Calculate total route distance
        total_distance = 0
        for i in range(len(coordinates) - 1):
            lat1, lon1 = coordinates[i]
            lat2, lon2 = coordinates[i + 1]
            total_distance += self._haversine_distance(lat1, lon1, lat2, lon2)

        self.logger.info(
            f"Added route line with {len(coordinates)} points. Total distance: {total_distance:.2f} km")

        # Update coverage data
        self.coverage["total_distance"] = total_distance

        return map_obj

    def _calculate_visible_range(self, point: Dict[str, float], fov: float, heading: float) -> List[Dict[str, float]]:
        """
        Calculate visible range coordinates for a point.

        Args:
            point: Point with lat/lng
            fov: Field of view in degrees
            heading: Camera heading in degrees

        Returns:
            List of polygon coordinates
        """
        # Constants
        r = 0.05  # Visible range in km

        # Get center point
        lat = point.get("latitude", 0)
        lng = point.get("longitude", 0)

        if lat == 0 or lng == 0:
            return []

        # Calculate visible range polygon
        heading_rad = math.radians(heading)
        fov_rad = math.radians(fov)

        # Center and two sides of the view triangle
        start_angle = heading_rad - fov_rad / 2
        end_angle = heading_rad + fov_rad / 2

        # Convert to x, y
        R = 6371.0  # Earth radius in km
        x_center = R * math.cos(math.radians(lat)) * \
            math.cos(math.radians(lng))
        y_center = R * math.cos(math.radians(lat)) * \
            math.sin(math.radians(lng))
        z_center = R * math.sin(math.radians(lat))

        # Create polygon points
        points = []
        points.append({"latitude": lat, "longitude": lng})  # Center point

        # Add points along the arc
        num_points = 10
        for i in range(num_points + 1):
            angle = start_angle + (end_angle - start_angle) * i / num_points
            # Convert back to lat/lng
            x = x_center + r * math.cos(angle)
            y = y_center + r * math.sin(angle)
            z = z_center

            new_lng = math.degrees(math.atan2(y, x))
            new_lat = math.degrees(math.asin(z / R))

            points.append({"latitude": new_lat, "longitude": new_lng})

        return points

    def add_coverage_polygons(self, map_obj: folium.Map,
                              image_points: List[Dict[str, Any]]) -> folium.Map:
        """
        Add coverage polygons for each image point.

        Args:
            map_obj: Folium map object
            image_points: List of image points with lat/lng and heading

        Returns:
            Updated map object
        """
        if not image_points:
            return map_obj

        # Create feature group for coverage
        coverage_group = folium.FeatureGroup(name="Image Coverage")

        # Add coverage polygon for each point
        for point in image_points:
            fov = point.get("fov", 55)  # Default 55 degrees if not specified
            heading = point.get("heading", 0)

            # Calculate visible range polygon
            polygon_points = self._calculate_visible_range(point, fov, heading)

            if polygon_points:
                # Extract coordinates in lat, lng format
                coordinates = [[p["latitude"], p["longitude"]]
                               for p in polygon_points]

                # Add polygon to map
                folium.Polygon(
                    locations=coordinates,
                    color=self.colors["coverage"],
                    fill=True,
                    fill_opacity=0.2,
                    weight=1
                ).add_to(coverage_group)

        # Add coverage group to map
        coverage_group.add_to(map_obj)

        self.logger.info(
            f"Added coverage polygons for {len(image_points)} image points")

        return map_obj

    def add_image_points(self, map_obj: folium.Map,
                         image_points: List[Dict[str, Any]],
                         detection_results: Optional[List[Dict[str, Any]]] = None) -> folium.Map:
        """
        Add markers for image acquisition points, color-coded by detection status.

        Args:
            map_obj: Folium map object
            image_points: List of image points with lat/lng
            detection_results: Optional list of detection results

        Returns:
            Updated map object
        """
        if not image_points:
            return map_obj

        # Create lookup for detection results by point index if available
        detection_lookup = {}
        if detection_results:
            for result in detection_results:
                params = result.get("image_params", {})
                point_index = params.get("point_index")
                if point_index is not None:
                    has_detections = len(result.get(
                        "detection_result", {}).get("detections", [])) > 0
                    detection_lookup[point_index] = {
                        "has_detections": has_detections,
                        "num_detections": len(result.get("detection_result", {}).get("detections", [])),
                        "result": result
                    }

        # Create feature group for image points
        points_group = folium.FeatureGroup(name="Image Points")
        detection_group = folium.FeatureGroup(name="Detection Points")

        # Add marker for each point
        for i, point in enumerate(image_points):
            lat = point.get("latitude")
            lng = point.get("longitude")

            if lat is not None and lng is not None and lat != 0 and lng != 0:
                # Check if this point has detection results
                point_index = point.get("point_index", i)
                has_detections = False
                detection_info = None

                if point_index in detection_lookup:
                    detection_info = detection_lookup[point_index]
                    has_detections = detection_info["has_detections"]

                # Create icon based on detection status
                if has_detections:
                    icon = folium.Icon(color="red", icon="camera")
                    target_group = detection_group
                else:
                    icon = folium.Icon(color="blue", icon="camera")
                    target_group = points_group

                # Create tooltip with point info
                tooltip = f"Point {point_index}<br>Heading: {point.get('heading', 0)}°"

                if detection_info:
                    tooltip += f"<br>Detections: {detection_info['num_detections']}"

                # Add marker
                folium.Marker(
                    location=[lat, lng],
                    icon=icon,
                    tooltip=tooltip
                ).add_to(target_group)

        # Add groups to map
        points_group.add_to(map_obj)
        detection_group.add_to(map_obj)

        self.logger.info(
            f"Added {len(image_points)} image point markers to map")

        return map_obj

    def add_detection_markers(self, map_obj: folium.Map,
                              detection_results: List[Dict[str, Any]]) -> folium.Map:
        """
        Add markers for sleeve detections.

        Args:
            map_obj: Folium map object
            detection_results: List of detection results

        Returns:
            Updated map object
        """
        if not detection_results:
            return map_obj

        # Create feature group for detections
        detection_group = folium.FeatureGroup(name="Sleeve Detections")

        # Track unique detections (avoid duplicates at same location)
        seen_locations = set()

        # Process each detection result
        for result in detection_results:
            params = result.get("image_params", {})
            detections = result.get(
                "detection_result", {}).get("detections", [])

            if not detections:
                continue

            # Get image point location
            lat = params.get("latitude")
            lng = params.get("longitude")

            if lat is None or lng is None or lat == 0 or lng == 0:
                continue

            # Skip if we've already seen a detection at this location
            location_key = f"{lat:.6f},{lng:.6f}"
            if location_key in seen_locations:
                continue

            seen_locations.add(location_key)

            # Create and add marker
            popup_html = f"""
            <div style="width:200px">
                <h4>Sleeve Detections</h4>
                <p>Number of sleeves: {len(detections)}</p>
                <p>Confidence: {max([d.get("confidence", 0) for d in detections]):.2f}</p>
                <p>Location: {lat:.6f}, {lng:.6f}</p>
            </div>
            """

            # Create an icon with marker
            folium.Marker(
                location=[lat, lng],
                icon=folium.Icon(color="purple", icon="bolt"),
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"Sleeves: {len(detections)}"
            ).add_to(detection_group)

        # Add detection group to map
        detection_group.add_to(map_obj)

        num_detections = len(seen_locations)
        self.logger.info(f"Added {num_detections} detection markers to map")

        return map_obj

    def calculate_route_coverage(self, route_points: List[Dict[str, float]],
                                 image_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate coverage statistics for the route.

        Args:
            route_points: List of route points with lat/lng
            image_points: List of image points with lat/lng and heading

        Returns:
            Dictionary with coverage statistics
        """
        if not route_points or not image_points:
            return {
                "total_distance": 0,
                "covered_distance": 0,
                "coverage_percentage": 0,
                "covered_segments": []
            }

        # Extract route coordinates
        route_coords = []
        for point in route_points:
            lat = point.get("latitude")
            lng = point.get("longitude")
            if lat is not None and lng is not None and lat != 0 and lng != 0:
                route_coords.append((lat, lng))

        if not route_coords:
            return {
                "total_distance": 0,
                "covered_distance": 0,
                "coverage_percentage": 0,
                "covered_segments": []
            }

        # Calculate total route distance
        total_distance = 0
        for i in range(len(route_coords) - 1):
            lat1, lon1 = route_coords[i]
            lat2, lon2 = route_coords[i + 1]
            total_distance += self._haversine_distance(lat1, lon1, lat2, lon2)

        # Track which segments are covered
        segment_coverage = [False] * (len(route_coords) - 1)
        covered_segments = []

        # For each image point, check which segments it covers
        visible_range = 0.05  # km
        for point in image_points:
            fov = point.get("fov", 55)  # Default 55 degrees if not specified
            heading = point.get("heading", 0)

            # Calculate visible range polygon
            polygon_points = self._calculate_visible_range(point, fov, heading)

            if not polygon_points:
                continue

            # Extract polygon coordinates
            polygon_coords = [(p["latitude"], p["longitude"])
                              for p in polygon_points]

            # Check each route segment
            for i in range(len(route_coords) - 1):
                if segment_coverage[i]:
                    continue  # Skip segments already covered

                segment_start = route_coords[i]
                segment_end = route_coords[i + 1]

                # Simple check: distance from image point to segment
                # This is a simplification; a more accurate approach would check
                # for intersection between the segment and the field-of-view polygon
                img_lat = point.get("latitude", 0)
                img_lng = point.get("longitude", 0)

                if img_lat == 0 or img_lng == 0:
                    continue

                # Check distances to segment endpoints
                d1 = self._haversine_distance(
                    img_lat, img_lng, segment_start[0], segment_start[1])
                d2 = self._haversine_distance(
                    img_lat, img_lng, segment_end[0], segment_end[1])

                # If either endpoint is within visible range, mark as covered
                if d1 <= visible_range or d2 <= visible_range:
                    segment_coverage[i] = True
                    covered_segments.append({
                        "segment_index": i,
                        "start": {"latitude": segment_start[0], "longitude": segment_start[1]},
                        "end": {"latitude": segment_end[0], "longitude": segment_end[1]}
                    })

        # Calculate covered distance
        covered_distance = 0
        for i, covered in enumerate(segment_coverage):
            if covered:
                lat1, lon1 = route_coords[i]
                lat2, lon2 = route_coords[i + 1]
                covered_distance += self._haversine_distance(
                    lat1, lon1, lat2, lon2)

        # Calculate coverage percentage
        coverage_percentage = (
            covered_distance / total_distance * 100) if total_distance > 0 else 0

        # Update coverage data
        self.coverage.update({
            "total_distance": total_distance,
            "covered_distance": covered_distance,
            "coverage_percentage": coverage_percentage,
            "covered_segments": covered_segments
        })

        self.logger.info(
            f"Route coverage: {coverage_percentage:.2f}% ({covered_distance:.2f}/{total_distance:.2f} km)")

        return self.coverage

    def add_coverage_stats(self, map_obj: folium.Map) -> folium.Map:
        """
        Add coverage statistics to the map as a control panel.

        Args:
            map_obj: Folium map object

        Returns:
            Updated map object
        """
        # Create HTML for coverage stats
        html = f"""
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 180px;
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white; padding: 10px;
                    opacity: 0.8;">
            <div style="font-weight: bold;">Coverage Statistics</div>
            <hr style="margin: 5px 0;">
            <div>Total distance: {self.coverage["total_distance"]:.2f} km</div>
            <div>Covered distance: {self.coverage["covered_distance"]:.2f} km</div>
            <div>Coverage: {self.coverage["coverage_percentage"]:.2f}%</div>
        </div>
        """

        # Add HTML to map
        map_obj.get_root().html.add_child(folium.Element(html))

        return map_obj

    def generate_map(self, route_points: List[Dict[str, float]],
                     image_points: List[Dict[str, Any]],
                     detection_results: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate an interactive map with route, coverage, and detections.

        Args:
            route_points: List of route points with lat/lng
            image_points: List of image points with lat/lng and heading
            detection_results: Optional list of detection results

        Returns:
            Path to saved HTML map file
        """
        self.logger.info("Generating coverage map")
        start_time = time.time()

        # Create base map
        map_obj = self.create_base_map(route_points)

        # Add route line
        map_obj = self.add_route_line(map_obj, route_points)

        # Add coverage polygons
        map_obj = self.add_coverage_polygons(map_obj, image_points)

        # Add image points
        map_obj = self.add_image_points(
            map_obj, image_points, detection_results)

        # Add detection markers if available
        if detection_results:
            map_obj = self.add_detection_markers(map_obj, detection_results)

        # Calculate route coverage
        self.calculate_route_coverage(route_points, image_points)

        # Add coverage statistics
        map_obj = self.add_coverage_stats(map_obj)

        # Save map to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.output_dir, f"coverage_map_{timestamp}.html")
        map_obj.save(output_path)

        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Map generated and saved to {output_path} in {elapsed_time:.2f}s")

        return output_path

    def generate_detection_heatmap(self, detection_results: List[Dict[str, Any]]) -> str:
        """
        Generate a heatmap of sleeve detections.

        Args:
            detection_results: List of detection results

        Returns:
            Path to saved heatmap image
        """
        if not detection_results:
            self.logger.warning("No detection results for heatmap")
            return None

        try:
            # Extract coordinates and confidences
            points = []
            confidences = []

            for result in detection_results:
                params = result.get("image_params", {})
                detections = result.get(
                    "detection_result", {}).get("detections", [])

                if not detections:
                    continue

                lat = params.get("latitude")
                lng = params.get("longitude")

                if lat is None or lng is None or lat == 0 or lng == 0:
                    continue

                # Add a point for each detection at this location
                for det in detections:
                    points.append((lat, lng))
                    confidences.append(det.get("confidence", 0.5))

            if not points:
                self.logger.warning("No valid detection points for heatmap")
                return None

            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame({
                "confidence": confidences,
                "geometry": [Point(lng, lat) for lat, lng in points]
            }, crs="EPSG:4326")

            # Convert to a projected CRS for better visual representation
            gdf = gdf.to_crs(epsg=3857)

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot detection points as a heatmap
            gdf.plot(ax=ax, column="confidence", cmap="hot",
                     markersize=20, alpha=0.6, legend=True)

            # Add basemap
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

            # Set title and labels
            ax.set_title("Sleeve Detection Heatmap", fontsize=16)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

            # Save figure
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.output_dir, f"detection_heatmap_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()

            self.logger.info(f"Heatmap generated and saved to {output_path}")
            return output_path

        except ImportError as e:
            self.logger.error(
                f"Heatmap generation failed: {e}. Make sure geopandas and contextily are installed.")
            return None
        except Exception as e:
            self.logger.error(f"Heatmap generation failed: {e}")
            return None

    def export_geojson(self, route_points: List[Dict[str, float]],
                       detection_results: List[Dict[str, Any]]) -> str:
        """
        Export route and detections as GeoJSON for use in other mapping tools.

        Args:
            route_points: List of route points with lat/lng
            detection_results: List of detection results

        Returns:
            Path to saved GeoJSON file
        """
        try:
            # Extract route coordinates
            route_coords = []
            for point in route_points:
                lat = point.get("latitude")
                lng = point.get("longitude")
                if lat is not None and lng is not None and lat != 0 and lng != 0:
                    # GeoJSON uses (lng, lat) order
                    route_coords.append((lng, lat))

            if not route_coords:
                self.logger.warning(
                    "No valid route coordinates for GeoJSON export")
                return None

            # Create route LineString
            route_line = LineString(route_coords)

            # Create GeoDataFrame for route
            route_gdf = gpd.GeoDataFrame({"name": ["Route"]}, geometry=[
                                         route_line], crs="EPSG:4326")

            # Extract detection points
            detection_data = []

            for result in detection_results:
                params = result.get("image_params", {})
                detections = result.get(
                    "detection_result", {}).get("detections", [])

                if not detections:
                    continue

                lat = params.get("latitude")
                lng = params.get("longitude")

                if lat is None or lng is None or lat == 0 or lng == 0:
                    continue

                # Add a point for each detection
                for det in detections:
                    detection_data.append({
                        "confidence": det.get("confidence", 0),
                        "heading": params.get("heading", 0),
                        "timestamp": params.get("timestamp", ""),
                        # GeoJSON uses (lng, lat) order
                        "geometry": Point(lng, lat)
                    })

            # Create GeoDataFrame for detections
            if detection_data:
                detection_gdf = gpd.GeoDataFrame(
                    detection_data, crs="EPSG:4326")

                # Combine route and detections
                combined_gdf = gpd.GeoDataFrame(
                    pd.concat([route_gdf, detection_gdf], ignore_index=True))
            else:
                combined_gdf = route_gdf

            # Save as GeoJSON
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.output_dir, f"sleeve_detection_{timestamp}.geojson")
            combined_gdf.to_file(output_path, driver="GeoJSON")

            self.logger.info(f"GeoJSON exported to {output_path}")
            return output_path

        except ImportError as e:
            self.logger.error(
                f"GeoJSON export failed: {e}. Make sure geopandas is installed.")
            return None
        except Exception as e:
            self.logger.error(f"GeoJSON export failed: {e}")
            return None
