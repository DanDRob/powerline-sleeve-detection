#!/usr/bin/env python3
import sys
from .cli import main
import os
from dotenv import load_dotenv

# Load environment variables at the very beginning
load_dotenv()

if __name__ == "__main__":
    main()
