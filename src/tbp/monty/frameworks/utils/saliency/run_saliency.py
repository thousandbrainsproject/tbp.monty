#!/usr/bin/env python3
"""
Simple runner script for saliency detection comparison.

This script provides an easy entry point to run saliency comparisons
on all .npy files in the standard directory structure.
"""

import sys
from pathlib import Path

# Add current directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from static_saliency import SaliencyComparator, create_default_config


def main():
    """Run saliency comparison on all .npy files."""
    try:
        print("Starting saliency detection comparison...")
        
        # Create configuration
        config = create_default_config()
        print(f"Data directory: {config.data_dir}")
        
        # Create and run comparator
        comparator = SaliencyComparator(config)
        comparator.run_comparison()
        
        print("Saliency comparison completed successfully!")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()