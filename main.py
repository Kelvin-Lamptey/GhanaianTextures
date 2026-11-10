import numpy as np
from PIL import Image
import os

def height_to_normal_map(height_image_path, output_path, strength=2.0, opengl_format=True):
    # 1. Load image and convert to 32-bit float grayscale [0.0, 1.0]
    img = Image.open(height_image_path).convert('L')
    height = np.asarray(img, dtype=np.float32) / 255.0

    # 2. Compute spatial gradients using Central Differences
    # np.gradient calculates (df/dx, df/dy) across array axes
    dy, dx = np.gradient(height)

    # 3. Construct 3D unnormalized normal components
    # Apply strength multiplier to accentuate displacement depth
    nx = -dx * strength
    ny = -dy * strength if opengl_format else dy * strength
    nz = np.ones_like(height)

    # 4. Normalize normal vectors across depth channel
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= norm
    ny /= norm
    nz /= norm

    # 5. Map vectors from [-1.0, 1.0] to [0, 255] RGB space
    r = ((nx * 0.5 + 0.5) * 255.0).astype(np.uint8)
    g = ((ny * 0.5 + 0.5) * 255.0).astype(np.uint8)
    b = ((nz * 0.5 + 0.5) * 255.0).astype(np.uint8)

    # Stack channels into RGB image array (Height, Width, 3)
    normal_map = np.stack([r, g, b], axis=-1)

    # 6. Save final normal map
    result = Image.fromarray(normal_map)
    result.save(output_path)
    print(f"Normal map generated successfully and saved to {output_path}")

# Example Execution
if __name__ == "__main__":
    for texture in os.listdir("./heights"):
        height_to_normal_map(os.path.join("heights", texture), f"{os.path.join('normals', os.path.splitext(texture)[0])}_normal.png", strength=3.0)

