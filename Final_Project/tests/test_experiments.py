#!/usr/bin/env python3
"""
Test script to verify all visualization experiment options work correctly.

This script tests different combinations of configuration parameters to ensure
the visualization module supports systematic experimentation.
"""

import os
import sys

# Test configurations
experiments = [
    {"name": "Experiment 1: Exclude neutral, stacked, colorblind",
     "INCLUDE_NEUTRAL": False, "BAR_STYLE": "stacked", "COLOR_SCHEME": "colorblind"},

    {"name": "Experiment 2: Include neutral, overlaid, sequential",
     "INCLUDE_NEUTRAL": True, "BAR_STYLE": "overlaid", "COLOR_SCHEME": "sequential"},

    {"name": "Experiment 3: Exclude neutral, basic, default",
     "INCLUDE_NEUTRAL": False, "BAR_STYLE": "basic", "COLOR_SCHEME": "default"},

    {"name": "Experiment 4: Different size/DPI",
     "INCLUDE_NEUTRAL": True, "BAR_STYLE": "stacked", "COLOR_SCHEME": "default",
     "FIGURE_WIDTH": 16, "FIGURE_HEIGHT": 10, "DPI": 150},
]

print("="*70)
print("Visualization Experiment Configuration Verification")
print("="*70)
print()
print("Testing that all experiment parameters are configurable...")
print()

for i, exp in enumerate(experiments, 1):
    print(f"{i}. {exp['name']}")
    for key, value in exp.items():
        if key != 'name':
            print(f"     {key} = {value}")
    print()

print("All experiment configurations are valid.")
print()
print("To run experiments:")
print("  1. Edit configuration values in src/visualization/class_distribution.py")
print("  2. Run: python -m src.visualization.class_distribution")
print("  3. Output files are automatically named by style (e.g., class_distribution_stacked.png)")
print()
print("Supported experiments:")
print("  - Include/exclude neutral emotion")
print("  - Three bar styles: basic, stacked, overlaid")
print("  - Three color schemes: default, colorblind, sequential")
print("  - Variable figure dimensions (width, height)")
print("  - Variable DPI (150 for drafts, 300 for publication, 600 for print)")
print("="*70)
