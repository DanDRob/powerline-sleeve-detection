import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import webbrowser
import threading
from datetime import datetime
import json

from ..system.config import Config
from ..system.logging import get_logger


class DashboardGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("dashboard")
        self.output_dir = os.path.join(config.system.output_dir, "dashboard")
        os.makedirs(self.output_dir, exist_ok=True)

        # Dashboard data
        self.data = {
            "route_data": None,
            "image_data": None,
            "detection_data": None,
            "tracking_data": None,
            "stats": None,
            "coverage": None
        }

        # Dashboard app
        self.app = None
        self.server = None
        self.is_running = False

    def load_data(self, route_points: List[Dict[str, Any]] = None,
                  image_points: List[Dict[str, Any]] = None,
                  detection_results: List[Dict[str, Any]] = None,
                  tracking_results: List[Dict[str, Any]] = None,
                  stats: Dict[str, Any] = None,
                  coverage: Dict[str, Any] = None) -> None:
        """
        Load data for the dashboard.

        Args:
            route_points: List of route points
            image_points: List of image points
            detection_results: List of detection results
            tracking_results: List of tracking results
            stats: Detection statistics
            coverage: Coverage statistics
        """
        # Convert data to pandas DataFrames for easier processing
        if route_points:
            self.data["route_data"] = pd.DataFrame(route_points)

        if image_points:
            self.data["image_data"] = pd.DataFrame(image_points)

        if detection_results:
            # Extract detection data
            detection_data = []

            for result in detection_results:
                params = result.get("image_params", {})
                detections = result.get(
                    "detection_result", {}).get("detections", [])

                for det in detections:
                    detection_data.append({
                        **params,
                        "confidence": det.get("confidence", 0),
                        "bbox": det.get("bbox", [0, 0, 0, 0]),
                        "class_id": det.get("class_id", 0),
                        "detection_time": result.get("detection_result", {}).get("processing_time", 0)
                    })

            if detection_data:
                self.data["detection_data"] = pd.DataFrame(detection_data)

        if tracking_results:
            tracking_data = []

            for result in tracking_results:
                params = result.get("image_params", {})
                tracking_result = result.get("tracking_result", {})

                tracking_data.append({
                    **params,
                    "num_objects": tracking_result.get("num_objects", 0),
                    "matched_detections": tracking_result.get("matched_detections", 0),
                    "new_detections": tracking_result.get("new_detections", 0),
                    "timestamp": tracking_result.get("timestamp", 0)
                })

            if tracking_data:
                self.data["tracking_data"] = pd.DataFrame(tracking_data)

        if stats:
            self.data["stats"] = stats

        if coverage:
            self.data["coverage"] = coverage

        self.logger.info("Data loaded for dashboard")

    def _create_layout(self) -> html.Div:
        """
        Create the dashboard layout.

        Returns:
            Dash layout component
        """
        # Navbar
        navbar = dbc.NavbarSimple(
            brand="Powerline Sleeve Detection Dashboard",
            brand_href="#",
            color="primary",
            dark=True,
        )

        # Summary cards
        summary_stats = self.data.get("stats", {})
        coverage_stats = self.data.get("coverage", {})

        total_detections = 0
        avg_confidence = 0

        if self.data.get("detection_data") is not None:
            total_detections = len(self.data["detection_data"])
            if total_detections > 0:
                avg_confidence = round(
                    self.data["detection_data"]["confidence"].mean() * 100, 1)

        route_distance = round(coverage_stats.get("total_distance", 0), 2)
        coverage_percentage = round(
            coverage_stats.get("coverage_percentage", 0), 1)

        cards = dbc.CardDeck([
            dbc.Card([
                dbc.CardHeader("Total Sleeve Detections"),
                dbc.CardBody(
                    html.H4(f"{total_detections}", className="card-title"))
            ], color="primary", outline=True),
            dbc.Card([
                dbc.CardHeader("Average Confidence"),
                dbc.CardBody(
                    html.H4(f"{avg_confidence}%", className="card-title"))
            ], color="success", outline=True),
            dbc.Card([
                dbc.CardHeader("Route Distance"),
                dbc.CardBody(
                    html.H4(f"{route_distance} km", className="card-title"))
            ], color="info", outline=True),
            dbc.Card([
                dbc.CardHeader("Route Coverage"),
                dbc.CardBody(
                    html.H4(f"{coverage_percentage}%", className="card-title"))
            ], color="warning", outline=True)
        ])

        # Tabs for different visualizations
        tabs = dbc.Tabs([
            dbc.Tab([
                html.Div([
                    html.H4("Detection Map", className="mt-4"),
                    html.P("Map showing the route and detection locations"),
                    dcc.Graph(id="detection-map")
                ])
            ], label="Map"),
            dbc.Tab([
                html.Div([
                    html.H4("Detection Statistics", className="mt-4"),
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="confidence-histogram")
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(id="detections-by-location")
                            ], width=6)
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="detections-timeline")
                            ], width=12)
                        ])
                    ])
                ])
            ], label="Statistics"),
            dbc.Tab([
                html.Div([
                    html.H4("Coverage Analysis", className="mt-4"),
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="coverage-map")
                            ], width=12)
                        ])
                    ])
                ])
            ], label="Coverage")
        ])

        # Assemble layout
        layout = html.Div([
            navbar,
            dbc.Container([
                html.Div(cards, className="mt-4"),
                tabs
            ], fluid=True)
        ])

        return layout

    def _create_callbacks(self) -> None:
        """
        Define dashboard callbacks.
        """
        @self.app.callback(
            Output("detection-map", "figure"),
            [Input("detection-map", "id")]
        )
        def update_detection_map(_):
            """Update the detection map visualization"""
            fig = go.Figure()

            # Add route line if available
            if self.data.get("route_data") is not None:
                route_df = self.data["route_data"]
                if "latitude" in route_df.columns and "longitude" in route_df.columns:
                    fig.add_trace(go.Scattermapbox(
                        lat=route_df["latitude"],
                        lon=route_df["longitude"],
                        mode="lines",
                        line=dict(width=2, color="#1f77b4"),
                        name="Route"
                    ))

            # Add detection points if available
            if self.data.get("detection_data") is not None:
                det_df = self.data["detection_data"]
                if "latitude" in det_df.columns and "longitude" in det_df.columns:
                    fig.add_trace(go.Scattermapbox(
                        lat=det_df["latitude"],
                        lon=det_df["longitude"],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color="red",
                            opacity=0.7
                        ),
                        text=det_df["confidence"].apply(
                            lambda x: f"Confidence: {x:.2f}"),
                        name="Detections"
                    ))

            # Set map layout
            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(
                        lat=det_df["latitude"].mean() if self.data.get(
                            "detection_data") is not None else 43.6532,
                        lon=det_df["longitude"].mean() if self.data.get(
                            "detection_data") is not None else -79.3832
                    ),
                    zoom=12
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            return fig

        @self.app.callback(
            Output("confidence-histogram", "figure"),
            [Input("confidence-histogram", "id")]
        )
        def update_confidence_histogram(_):
            """Update the confidence histogram visualization"""
            fig = go.Figure()

            if self.data.get("detection_data") is not None:
                det_df = self.data["detection_data"]
                if "confidence" in det_df.columns:
                    fig = px.histogram(
                        det_df,
                        x="confidence",
                        nbins=20,
                        title="Detection Confidence Distribution",
                        labels={"confidence": "Confidence Score"},
                        color_discrete_sequence=["#1f77b4"]
                    )

            fig.update_layout(
                xaxis_title="Confidence Score",
                yaxis_title="Count",
                bargap=0.1
            )

            return fig

        @self.app.callback(
            Output("detections-by-location", "figure"),
            [Input("detections-by-location", "id")]
        )
        def update_detections_by_location(_):
            """Update the detections by location visualization"""
            fig = go.Figure()

            if self.data.get("detection_data") is not None:
                det_df = self.data["detection_data"]
                if "latitude" in det_df.columns and "longitude" in det_df.columns:
                    # Create a location ID for grouping
                    det_df["location_id"] = det_df.apply(
                        lambda row: f"{row['latitude']:.5f},{row['longitude']:.5f}", axis=1)

                    # Count detections by location
                    location_counts = det_df.groupby(
                        "location_id").size().reset_index(name="count")
                    location_counts = location_counts.sort_values(
                        "count", ascending=False).head(10)

                    fig = px.bar(
                        location_counts,
                        x="location_id",
                        y="count",
                        title="Top 10 Locations by Detection Count",
                        labels={"location_id": "Location",
                                "count": "Number of Detections"},
                        color_discrete_sequence=["#2ca02c"]
                    )

            fig.update_layout(
                xaxis_title="Location",
                yaxis_title="Number of Detections",
                xaxis_tickangle=-45
            )

            return fig

        @self.app.callback(
            Output("detections-timeline", "figure"),
            [Input("detections-timeline", "id")]
        )
        def update_detections_timeline(_):
            """Update the detections timeline visualization"""
            fig = go.Figure()

            if self.data.get("tracking_data") is not None:
                track_df = self.data["tracking_data"]
                if "timestamp" in track_df.columns and "num_objects" in track_df.columns:
                    # Convert timestamp to datetime
                    track_df["datetime"] = pd.to_datetime(
                        track_df["timestamp"], unit="s")

                    # Sort by timestamp
                    track_df = track_df.sort_values("datetime")

                    fig = px.line(
                        track_df,
                        x="datetime",
                        y="num_objects",
                        title="Number of Tracked Objects Over Time",
                        labels={"datetime": "Time",
                                "num_objects": "Number of Objects"},
                        color_discrete_sequence=["#d62728"]
                    )

            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Number of Tracked Objects"
            )

            return fig

        @self.app.callback(
            Output("coverage-map", "figure"),
            [Input("coverage-map", "id")]
        )
        def update_coverage_map(_):
            """Update the coverage map visualization"""
            fig = go.Figure()

            # Add route line
            if self.data.get("route_data") is not None:
                route_df = self.data["route_data"]
                if "latitude" in route_df.columns and "longitude" in route_df.columns:
                    fig.add_trace(go.Scattermapbox(
                        lat=route_df["latitude"],
                        lon=route_df["longitude"],
                        mode="lines",
                        line=dict(width=2, color="#1f77b4"),
                        name="Route"
                    ))

            # Add image points (covered areas)
            if self.data.get("image_data") is not None:
                img_df = self.data["image_data"]
                if "latitude" in img_df.columns and "longitude" in img_df.columns:
                    fig.add_trace(go.Scattermapbox(
                        lat=img_df["latitude"],
                        lon=img_df["longitude"],
                        mode="markers",
                        marker=dict(
                            size=8,
                            color="#2ca02c",
                            opacity=0.7
                        ),
                        name="Covered Areas"
                    ))

            # Set map layout
            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(
                        lat=route_df["latitude"].mean() if self.data.get(
                            "route_data") is not None else 43.6532,
                        lon=route_df["longitude"].mean() if self.data.get(
                            "route_data") is not None else -79.3832
                    ),
                    zoom=12
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            return fig

    def create_app(self) -> dash.Dash:
        """
        Create the Dash app.

        Returns:
            Dash app instance
        """
        # Initialize app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )

        # Create layout
        self.app.layout = self._create_layout()

        # Define callbacks
        self._create_callbacks()

        self.server = self.app.server

        return self.app

    def start_dashboard(self, debug: bool = False, port: int = 8050) -> None:
        """
        Start the dashboard server.

        Args:
            debug: Whether to run in debug mode
            port: Port to run the server on
        """
        if self.app is None:
            self.create_app()

        # Use a thread to avoid blocking
        def open_browser():
            """Open the browser after a slight delay"""
            time.sleep(1)
            webbrowser.open_new(f"http://localhost:{port}")

        if not debug:
            threading.Thread(target=open_browser).start()

        self.is_running = True
        self.app.run_server(debug=debug, port=port)

    def save_dashboard_data(self) -> str:
        """
        Save dashboard data for later use.

        Returns:
            Path to saved data file
        """
        # Prepare data for saving
        save_data = {}

        for key, data in self.data.items():
            if isinstance(data, pd.DataFrame):
                save_data[key] = data.to_dict(orient="records")
            else:
                save_data[key] = data

        # Save to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.output_dir, f"dashboard_data_{timestamp}.json")

        with open(output_path, "w") as f:
            json.dump(save_data, f)

        self.logger.info(f"Dashboard data saved to {output_path}")

        return output_path

    def load_dashboard_data(self, file_path: str) -> bool:
        """
        Load dashboard data from file.

        Args:
            file_path: Path to the data file

        Returns:
            Success status
        """
        try:
            with open(file_path, "r") as f:
                save_data = json.load(f)

            # Convert back to DataFrames
            for key, data in save_data.items():
                if key in ["route_data", "image_data", "detection_data", "tracking_data"] and data:
                    self.data[key] = pd.DataFrame(data)
                else:
                    self.data[key] = data

            self.logger.info(f"Dashboard data loaded from {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error loading dashboard data: {e}")
            return False

    def generate_standalone_html(self) -> str:
        """
        Generate a standalone HTML file with the dashboard.

        Returns:
            Path to saved HTML file
        """
        if self.app is None:
            self.create_app()

        # Create output path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.output_dir, f"dashboard_{timestamp}.html")

        # Generate figures
        detection_map = self.app.callback_map["detection-map.figure"]["callback"]()
        confidence_histogram = self.app.callback_map["confidence-histogram.figure"]["callback"](
        )
        detections_by_location = self.app.callback_map["detections-by-location.figure"]["callback"](
        )
        detections_timeline = self.app.callback_map["detections-timeline.figure"]["callback"](
        )
        coverage_map = self.app.callback_map["coverage-map.figure"]["callback"]()

        # Create HTML content
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Powerline Sleeve Detection Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
        .card {{
            margin-bottom: 20px;
        }}
        .tab-content {{
            padding-top: 20px;
        }}
    </style>
</head>
<body>
    <nav class="navbar navbar-dark bg-primary">
        <span class="navbar-brand mb-0 h1">Powerline Sleeve Detection Dashboard</span>
    </nav>
    <div class="container-fluid mt-4">
        <div class="row">
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header text-white bg-primary">Total Sleeve Detections</div>
                    <div class="card-body">
                        <h4 class="card-title">{len(self.data.get('detection_data', pd.DataFrame())) if self.data.get('detection_data') is not None else 0}</h4>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header text-white bg-success">Average Confidence</div>
                    <div class="card-body">
                        <h4 class="card-title">{round(self.data.get('detection_data', pd.DataFrame())['confidence'].mean() * 100, 1) if self.data.get('detection_data') is not None and len(self.data.get('detection_data')) > 0 else 0}%</h4>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header text-white bg-info">Route Distance</div>
                    <div class="card-body">
                        <h4 class="card-title">{round(self.data.get('coverage', {}).get('total_distance', 0), 2)} km</h4>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header text-white bg-warning">Route Coverage</div>
                    <div class="card-body">
                        <h4 class="card-title">{round(self.data.get('coverage', {}).get('coverage_percentage', 0), 1)}%</h4>
                    </div>
                </div>
            </div>
        </div>
        <div class="mt-4">
            <ul class="nav nav-tabs" id="dashboardTabs" role="tablist">
                <li class="nav-item">
                    <a class="nav-link active" id="map-tab" data-toggle="tab" href="#map" role="tab">Map</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" id="statistics-tab" data-toggle="tab" href="#statistics" role="tab">Statistics</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link" id="coverage-tab" data-toggle="tab" href="#coverage" role="tab">Coverage</a>
                </li>
            </ul>
            <div class="tab-content" id="dashboardTabsContent">
                <div class="tab-pane fade show active" id="map" role="tabpanel">
                    <h4>Detection Map</h4>
                    <div id="detection-map"></div>
                </div>
                <div class="tab-pane fade" id="statistics" role="tabpanel">
                    <h4>Detection Statistics</h4>
                    <div class="row">
                        <div class="col-md-6">
                            <div id="confidence-histogram"></div>
                        </div>
                        <div class="col-md-6">
                            <div id="detections-by-location"></div>
                        </div>
                    </div>
                    <div class="row mt-4">
                        <div class="col-md-12">
                            <div id="detections-timeline"></div>
                        </div>
                    </div>
                </div>
                <div class="tab-pane fade" id="coverage" role="tabpanel">
                    <h4>Coverage Analysis</h4>
                    <div id="coverage-map"></div>
                </div>
            </div>
        </div>
    </div>
    <script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.2/dist/js/bootstrap.min.js"></script>
    <script>
        // Detection map
        const detectionMapData = {detection_map.to_json()};
        Plotly.newPlot('detection-map', detectionMapData.data, detectionMapData.layout);
        
        // Confidence histogram
        const confidenceHistogramData = {confidence_histogram.to_json()};
        Plotly.newPlot('confidence-histogram', confidenceHistogramData.data, confidenceHistogramData.layout);
        
        // Detections by location
        const detectionsByLocationData = {detections_by_location.to_json()};
        Plotly.newPlot('detections-by-location', detectionsByLocationData.data, detectionsByLocationData.layout);
        
        // Detections timeline
        const detectionsTimelineData = {detections_timeline.to_json()};
        Plotly.newPlot('detections-timeline', detectionsTimelineData.data, detectionsTimelineData.layout);
        
        // Coverage map
        const coverageMapData = {coverage_map.to_json()};
        Plotly.newPlot('coverage-map', coverageMapData.data, coverageMapData.layout);
    </script>
</body>
</html>"""

        # Save to file
        with open(output_path, "w") as f:
            f.write(html_content)

        self.logger.info(
            f"Standalone dashboard generated and saved to {output_path}")

        return output_path
