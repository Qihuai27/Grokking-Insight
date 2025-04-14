import os
import random
from PIL import Image, ImageDraw
from math import cos, sin, pi

# Shape dictionary: key is number of sides, value is the weight
shapes = {i: i for i in range(13)}
valid_shapes = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Output directories
image_dir = "shape-4-tan-white/out-shape-img"
label_dir = "shape-4-tan-white/out-shape-txt"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

def draw_polygon(draw, center, sides, radius, outline='black', fill='white'):
    """Draw a regular polygon with center lines on the canvas."""
    if sides < 3:
        return
    max_radius = min(center[0], center[1], 112 - center[0], 112 - center[1])
    adjusted_radius = min(radius, max_radius)
    angle = 2 * pi / sides
    points = [
        (center[0] + adjusted_radius * cos(i * angle),
         center[1] + adjusted_radius * sin(i * angle)) for i in range(sides)
    ]
    draw.polygon(points, outline=outline, fill=fill)
    for point in points:
        draw.line([center, point], fill=outline)

def generate_quad_image(index, shape_selection):
    """Generate and save a 2x2 grid image with selected shapes."""
    img = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(img)
    total_weight = sum(shapes[s] for s in shape_selection)

    for i, sides in enumerate(shape_selection):
        x_offset = (i % 2) * 112
        y_offset = (i // 2) * 112
        if sides != 0:
            draw_polygon(draw, (x_offset + 56, y_offset + 56), sides, radius=46)

    img.save(os.path.join(image_dir, f"{index}.jpg"))
    with open(os.path.join(label_dir, f"{index}.txt"), 'w') as f:
        f.write(f"Total Weight: {total_weight}\n")

def generate_all_combinations(valid_set):
    """Generate all 4-shape combinations from the valid set."""
    count = 0
    for a in valid_set:
        for b in valid_set:
            for c in valid_set:
                for d in valid_set:
                    count += 1
                    print(f"Generating image {count}")
                    generate_quad_image(count, [a, b, c, d])

if __name__ == "__main__":
    generate_all_combinations(valid_shapes)
