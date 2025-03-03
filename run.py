#!/usr/bin/env python3
import sys
from powerline_sleeve_detection.cli import main
from dotenv import load_dotenv

# Load environment variables at the very beginning
load_dotenv()

if __name__ == "__main__":
    # If no arguments are provided, show a helpful message
    if len(sys.argv) == 1:
        print("Powerline Sleeve Detection Utility")
        print("\nUsage: python run.py COMMAND [OPTIONS]")
        print("\nAvailable commands:")
        print("  process   Process routes and detect sleeves")
        print("  plan      Plan routes for data acquisition")
        print("  train     Train and evaluate sleeve detection models")
        print("  two-stage Two-stage detection for powerlines and sleeves")
        print("\nFor more information, use: python run.py COMMAND --help")
        sys.exit(1)

    # Otherwise, run the CLI as normal
    main()
{}
