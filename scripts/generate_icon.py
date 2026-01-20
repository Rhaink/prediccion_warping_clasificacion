#!/usr/bin/env python3
"""
Generate a simple icon for the COVID-19 Detection System executable.

This creates a basic medical-themed icon with a lung/medical cross symbol.
For production, consider using a professional icon designer or tool.

Usage:
    python scripts/generate_icon.py

Output:
    assets/covid_icon.ico (multi-resolution Windows icon)
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def create_icon():
    """Create a simple medical-themed icon."""
    # Create assets directory if it doesn't exist
    assets_dir = Path(__file__).parent.parent / 'assets'
    assets_dir.mkdir(exist_ok=True)

    # Icon sizes for Windows (multiple resolutions)
    sizes = [16, 32, 48, 64, 128, 256]
    images = []

    for size in sizes:
        # Create image with transparent background
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Colors
        bg_color = (41, 128, 185, 255)  # Medical blue
        cross_color = (255, 255, 255, 255)  # White

        # Draw circular background
        margin = size // 10
        draw.ellipse(
            [margin, margin, size - margin, size - margin],
            fill=bg_color,
            outline=(52, 73, 94, 255),
            width=max(1, size // 32)
        )

        # Draw medical cross
        cross_width = size // 4
        cross_height = size // 2
        center_x = size // 2
        center_y = size // 2

        # Horizontal bar
        draw.rectangle(
            [
                center_x - cross_height // 2,
                center_y - cross_width // 2,
                center_x + cross_height // 2,
                center_y + cross_width // 2
            ],
            fill=cross_color
        )

        # Vertical bar
        draw.rectangle(
            [
                center_x - cross_width // 2,
                center_y - cross_height // 2,
                center_x + cross_width // 2,
                center_y + cross_height // 2
            ],
            fill=cross_color
        )

        images.append(img)

    # Save as ICO file with multiple resolutions
    output_path = assets_dir / 'covid_icon.ico'
    images[0].save(
        output_path,
        format='ICO',
        sizes=[(s, s) for s in sizes]
    )

    print(f"✓ Icon created: {output_path}")
    print(f"  Resolutions: {', '.join(f'{s}x{s}' for s in sizes)}")

    # Also save a PNG version for documentation
    png_path = assets_dir / 'covid_icon.png'
    images[-1].save(png_path, format='PNG')
    print(f"✓ PNG version: {png_path}")

    return output_path


if __name__ == '__main__':
    try:
        create_icon()
    except Exception as e:
        print(f"Error creating icon: {e}")
        print("\nNote: Icon is optional. Build will work without it.")
        print("For a professional icon, consider:")
        print("  - Hiring a designer on Fiverr/Upwork")
        print("  - Using icon generation tools like IconGenerator")
        print("  - Downloading from icon libraries (with proper license)")
