import os
import argparse
import asyncio
import json
import sys
import logging
from typing import Dict, List, Any, Optional

from .system.config import Config
from .system.logging import setup_logging
from .processing.batch_processor import BatchProcessor
from dotenv import load_dotenv

# Load environment variables at the very beginning
load_dotenv()


def load_config(config_path: str) -> Config:
    """Load configuration from file."""
    try:
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)

        config = Config.from_yaml(config_path)
        validation_errors = config.validate()

        if validation_errors:
            print("Config validation errors:")
            for error in validation_errors:
                print(f"- {error}")
            sys.exit(1)

        return config

    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)


def setup_environment(config: Config):
    """Set up the environment for processing."""
    # Create output directories
    config.create_output_dirs()

    # Configure logging
    log_dir = os.path.join(config.system.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "powerline_detector.log")

    log_level = "DEBUG" if config.system.debug else "INFO"
    setup_logging(log_level, log_file)

    # Set API key from environment variable if not in config
    if not config.api.key and "GOOGLE_API_KEY" in os.environ:
        config.api.key = os.environ["GOOGLE_API_KEY"]


async def process_single_route(config: Config, route_id: str, start_location: str, end_location: str):
    """Process a single route."""
    processor = BatchProcessor(config)

    print(f"Processing route {route_id}: {start_location} to {end_location}")

    result = await processor.process_route(
        route_id=route_id,
        start_location=start_location,
        end_location=end_location,
        save_images=True,
        draw_detections=True,
        generate_map=True
    )

    if result["success"]:
        print(f"Route {route_id} processed successfully:")
        print(f"- Total images: {result['stats']['total_images']}")
        print(f"- Total detections: {result['stats']['total_detections']}")
        print(f"- Processing time: {result['stats']['duration']:.2f}s")
    else:
        print(
            f"Route {route_id} processing failed: {result.get('error', 'Unknown error')}")

    # Export results
    output_dir = os.path.join(config.system.output_dir,
                              "batch_results", f"route_{route_id}")
    print(f"Results saved to: {output_dir}")


async def process_route_batch(config: Config, csv_file: str, parallel: bool = True,
                              max_concurrent: int = 2, subset: int = None):
    """Process a batch of routes from a CSV file."""
    processor = BatchProcessor(config)

    # Load routes from CSV
    routes = processor.load_routes_from_csv(csv_file)
    if not routes:
        print(f"No valid routes found in {csv_file}")
        return

    # Create validation subset if requested
    if subset:
        print(f"Creating validation subset with {subset} routes")
        subset_csv = os.path.join(
            config.system.output_dir, "validation_subset.csv")
        routes = processor.create_route_subset_for_validation(
            routes, subset, subset_csv)
        print(f"Validation subset saved to {subset_csv}")

    total_routes = len(routes)
    print(
        f"Processing {total_routes} routes {'in parallel' if parallel else 'sequentially'}")

    # Process routes
    if parallel:
        result = await processor.process_routes_parallel(routes, max_concurrent)
    else:
        result = await processor.process_routes_sequential(routes)

    # Print summary
    summary = result["summary"]
    print("\nBatch Processing Summary:")
    print(f"- Total routes: {summary['total_routes']}")
    print(f"- Successful routes: {summary['successful_routes']}")
    print(f"- Failed routes: {summary['failed_routes']}")
    print(f"- Total images: {summary['total_images']}")
    print(f"- Total detections: {summary['total_detections']}")
    print(f"- Processing time: {summary['duration_seconds']:.2f}s")

    # Export combined results
    combined_results = processor.export_combined_results()
    if combined_results:
        print(f"Combined detection results exported to: {combined_results}")

    # Print output location
    print(f"Batch processing results saved to: {processor.output_dir}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Powerline Sleeve Detection System")

    # Config options
    parser.add_argument("--config", "-c", type=str, default="config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")

    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Single route processing
    route_parser = subparsers.add_parser(
        "route", help="Process a single route")
    route_parser.add_argument("--id", type=str, required=True,
                              help="Route identifier")
    route_parser.add_argument("--start", type=str, required=True,
                              help="Start location (address or lat,lng)")
    route_parser.add_argument("--end", type=str, required=True,
                              help="End location (address or lat,lng)")

    # Batch processing
    batch_parser = subparsers.add_parser(
        "batch", help="Process multiple routes from CSV")
    batch_parser.add_argument("--csv", type=str, required=True,
                              help="CSV file with route definitions")
    batch_parser.add_argument("--parallel", action="store_true",
                              help="Process routes in parallel")
    batch_parser.add_argument("--max-concurrent", type=int, default=2,
                              help="Maximum number of concurrent routes when using parallel processing")
    batch_parser.add_argument("--subset", type=int,
                              help="Create and process a validation subset with N routes")

    # Parse arguments
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override debug setting if specified
    if args.debug:
        config.system.debug = True

    # Set up environment
    setup_environment(config)

    # Process command
    if args.command == "route":
        asyncio.run(process_single_route(
            config, args.id, args.start, args.end))
    elif args.command == "batch":
        asyncio.run(process_route_batch(config, args.csv,
                    args.parallel, args.max_concurrent, args.subset))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
