#!/usr/bin/env python3
"""Creates GIFs from LAF image sets in scripts/laf_images/"""

import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def draw_label(img: Image.Image, text: str) -> Image.Image:
    """Draw a text label at the bottom of the image."""
    draw = ImageDraw.Draw(img)

    # Try to use a monospace font, fall back to default
    font_size = 14
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Position at bottom center with padding
    padding = 4
    x = (img.width - text_width) // 2
    y = img.height - text_height - padding * 2

    # Draw semi-transparent background
    draw.rectangle(
        [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
        fill=(0, 0, 0, 180),
    )

    # Draw text
    draw.text((x, y), text, fill=(255, 255, 255), font=font)

    return img


def main():
    script_dir = Path(__file__).parent
    input_dir = script_dir / "laf_images"
    output_dir = script_dir / "laf_gifs"

    # Duration per frame in ms
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    # Output width (height scales proportionally)
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 640

    output_dir.mkdir(exist_ok=True)

    # Find all jpeg files and group by prefix
    pattern = re.compile(r"^(.+-\d+)-(\d{3})-z.*\.jpeg$")
    groups: dict[str, list[tuple[int, Path]]] = {}

    for f in input_dir.glob("*.jpeg"):
        m = pattern.match(f.name)
        if m:
            prefix = m.group(1)
            frame_num = int(m.group(2))
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append((frame_num, f))

    if not groups:
        print(f"No JPEG files found in {input_dir}")
        sys.exit(1)

    for prefix in sorted(groups.keys()):
        print(f"Creating GIF for {prefix}...")

        # Sort by frame number
        files = [f for _, f in sorted(groups[prefix])]

        # Load and resize frames
        frames: list[Image.Image] = []
        for f in files:
            img = Image.open(f)
            # Calculate new height maintaining aspect ratio
            ratio = width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((width, new_height), Image.Resampling.LANCZOS)
            # Convert to RGB for drawing (if grayscale)
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Draw filename label
            img = draw_label(img, f.name)
            # Convert to palette mode for GIF
            img = img.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
            frames.append(img)

        # Save as GIF
        output_path = output_dir / f"{prefix}.gif"
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )
        print(f"  -> {output_path}")

    print(f"Done! GIFs saved to {output_dir}")


if __name__ == "__main__":
    main()
