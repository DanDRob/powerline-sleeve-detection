#!/usr/bin/env python3
import os
import sys
import argparse
from powerline_sleeve_detection.system.config import Config
from powerline_sleeve_detection.system.logging import setup_logging


def test_config():
    print("Testing config loading...")
    try:
        config = Config.from_yaml("config.yaml")
        print("✅ Config loaded successfully")

        # Test config validation
        errors = config.validate()
        if errors:
            print(f"❌ Config validation failed with errors: {errors}")
        else:
            print("✅ Config validation passed")

        # Test config to_dict and saving
        config_dict = config.to_dict()
        print(
            f"✅ Config converted to dictionary with {len(config_dict)} sections")

        return config
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return None


def test_logging(config):
    print("\nTesting logging...")
    try:
        log_level = "DEBUG" if config.system.debug else "INFO"
        log_file = os.path.join(config.system.output_dir, "logs", "test.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logger = setup_logging(log_level, log_file)
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message (don't worry, this is just a test)")
        print("✅ Logging is working")
    except Exception as e:
        print(f"❌ Logging setup failed: {e}")


def test_directories(config):
    print("\nTesting directory creation...")
    try:
        config.create_output_dirs()
        print(f"✅ Output directories created in {config.system.output_dir}")
    except Exception as e:
        print(f"❌ Directory creation failed: {e}")


def run_tests():
    print("Starting system tests...")

    # Test config loading
    config = test_config()
    if not config:
        print("❌ Cannot continue without valid config")
        return

    # Test logging
    test_logging(config)

    # Test directory creation
    test_directories(config)

    print("\nTests completed!")


if __name__ == "__main__":
    run_tests()
