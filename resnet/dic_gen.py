import os
import random
from PIL import Image, ImageDraw
from math import cos, sin, pi
import matplotlib.pyplot as plt

# Define shape dictionary: sides <= 2 are grouped into class 1, total of 11 classes
shapes = {i if i > 2 else 1: i for i in range(1, 13)}
shapes = {k: v for k, v in shapes.items() if k != 0}

# Define weights: shape 1 has weight 0, others have weight = sides - 1
weights = {1: 0}
weights.update({s: s - 1 for s in range(3, 13)})

def draw_polygon(draw, center, sides, radius, outline='black', fill='white'):
    if sides < 3:
        return
    max_radius = min(center[0], center[1], 224 - center[0], 224 - center[1])
    adjusted_radius = min(radius, max_radius)
    angle = 2 * pi / sides
    points = [
        (center[0] + adjusted_radius * cos(i * angle),
         center[1] + adjusted_radius * sin(i * angle)) 
        for i in range(sides)
    ]
    draw.polygon(points, outline=outline, fill=fill)
    for point in points:
        draw.line([center, point], fill=outline)

def generate_single_shape_image(sides):
    img = Image.new('RGB', (224, 224), color='white')
    draw = ImageDraw.Draw(img)
    center = (112, 112)
    radius = 90
    draw_polygon(draw, center, sides, radius)
    return img

def visualize_shapes_dictionary(save_path="shapes_grid_with_weights.png"):
    grid_size = (3, 4)
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 9))
    axes = axes.ravel()

    for i, (sides, _) in enumerate(shapes.items()):
        img = generate_single_shape_image(sides)
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(f'Shape {sides} (Weight: {weights[sides]})', fontsize=16, fontweight='bold')
        ax.axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    visualize_shapes_dictionary()
