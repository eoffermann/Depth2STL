import argparse
import numpy as np
from PIL import Image
import trimesh
import os
import tempfile
import gradio as gr


def load_image(path: str, invert: bool = False) -> np.ndarray:
    """Load an image and convert it to a grayscale numpy array.

    Args:
        path (str): Path to the image file.
        invert (bool): Whether to invert the grayscale values.

    Returns:
        np.ndarray: 2D array of grayscale values normalized to [0, 1].
    """
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return 1.0 - arr if invert else arr

def generate_mesh(
    image_array: np.ndarray, 
    width_cm: float, 
    max_depth_cm: float, 
    min_depth_cm: float, 
    step: int
) -> trimesh.Trimesh:
    """
    Generate a watertight 3D mesh from a grayscale image array.

    Args:
        image_array (np.ndarray): 2D normalized grayscale array (0..1).
        width_cm (float): Target width in cm (x-axis).
        max_depth_cm (float): Maximum height (z-axis, "peaks").
        min_depth_cm (float): Minimum height (z-axis, "valleys").
        step (int): Downsample step in pixels.

    Returns:
        trimesh.Trimesh: Watertight mesh.
    """
    img = image_array[::step, ::step]
    height_px, width_px = img.shape

    scale_x = width_cm / width_px
    scale_y = scale_x  # preserve aspect ratio
    scale_z = max_depth_cm - min_depth_cm

    # Top surface vertices
    vertices = []
    for y in range(height_px):
        for x in range(width_px):
            z = min_depth_cm + img[y, x] * scale_z
            vertices.append([x * scale_x, y * scale_y, z])
    vertices = np.array(vertices)

    # Helper for (x, y) to index
    def vidx(x, y):
        return y * width_px + x

    faces = []
    # Top surface faces (counter-clockwise winding)
    for y in range(height_px - 1):
        for x in range(width_px - 1):
            i0 = vidx(x, y)
            i1 = vidx(x + 1, y)
            i2 = vidx(x, y + 1)
            i3 = vidx(x + 1, y + 1)
            faces.append([i0, i1, i2])
            faces.append([i1, i3, i2])

    # Bottom vertices (flat base)
    base_z = min_depth_cm
    base_vertices = []
    for y in range(height_px):
        for x in range(width_px):
            base_vertices.append([x * scale_x, y * scale_y, base_z])
    base_vertices = np.array(base_vertices)
    base_offset = len(vertices)
    vertices = np.vstack([vertices, base_vertices])

    # Bottom faces (mirror top, but reversed winding)
    for y in range(height_px - 1):
        for x in range(width_px - 1):
            i0 = base_offset + vidx(x, y)
            i1 = base_offset + vidx(x + 1, y)
            i2 = base_offset + vidx(x, y + 1)
            i3 = base_offset + vidx(x + 1, y + 1)
            faces.append([i0, i2, i1])  # reversed winding
            faces.append([i1, i2, i3])

    # Sides (edges: left, right, top, bottom)
    def side_quad(top0, top1, bot0, bot1):
        # Always two triangles per quad
        faces.append([top0, top1, bot0])
        faces.append([top1, bot1, bot0])

    # Left and right sides
    for y in range(height_px - 1):
        # Left
        top0 = vidx(0, y)
        top1 = vidx(0, y + 1)
        bot0 = base_offset + vidx(0, y)
        bot1 = base_offset + vidx(0, y + 1)
        side_quad(top0, top1, bot0, bot1)
        # Right
        top0 = vidx(width_px - 1, y)
        top1 = vidx(width_px - 1, y + 1)
        bot0 = base_offset + vidx(width_px - 1, y)
        bot1 = base_offset + vidx(width_px - 1, y + 1)
        side_quad(top1, top0, bot1, bot0)  # flip winding to face outward

    # Top and bottom sides
    for x in range(width_px - 1):
        # Top
        top0 = vidx(x, 0)
        top1 = vidx(x + 1, 0)
        bot0 = base_offset + vidx(x, 0)
        bot1 = base_offset + vidx(x + 1, 0)
        side_quad(top1, top0, bot1, bot0)
        # Bottom
        top0 = vidx(x, height_px - 1)
        top1 = vidx(x + 1, height_px - 1)
        bot0 = base_offset + vidx(x, height_px - 1)
        bot1 = base_offset + vidx(x + 1, height_px - 1)
        side_quad(top0, top1, bot0, bot1)

    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)



def save_mesh(mesh: trimesh.Trimesh, output_path: str):
    """Save the mesh to an STL file.

    Args:
        mesh (trimesh.Trimesh): Mesh to save.
        output_path (str): Path to output STL file.
    """
    mesh.export(output_path)


def launch_gui():
    """Launch the Gradio GUI for interactive STL generation."""
    def process_gui(image, width, depth, min_depth, step, invert):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image.save(tmp.name)
            img_array = load_image(tmp.name, invert=invert)
        mesh = generate_mesh(img_array, width, depth, min_depth, step)
        tmp_stl = tempfile.NamedTemporaryFile(delete=False, suffix=".stl")
        save_mesh(mesh, tmp_stl.name)
        return tmp_stl.name

    inputs = [
        gr.Image(type="pil", label="Grayscale Depth Map (PNG/JPG)"),
        gr.Number(label="Width (cm)", value=10.0),
        gr.Number(label="Max Depth (cm)", value=1.0),
        gr.Number(label="Min Depth (cm)", value=0.1),
        gr.Slider(1, 100, value=10, label="Step Size (pixels)"),
        gr.Checkbox(label="Invert Grayscale")
    ]

    outputs = gr.Model3D(label="STL Preview", clear_color="#ffffff")

    gr.Interface(
        fn=process_gui,
        inputs=inputs,
        outputs=outputs,
        title="Image to STL Generator",
        allow_flagging="never"
    ).launch(inbrowser=True)


def main():
    parser = argparse.ArgumentParser(description="Convert grayscale image to STL depth map")
    parser.add_argument("-p", "--path", type=str, help="Path to input PNG/JPG image")
    parser.add_argument("-w", "--width", type=float, help="Width in cm")
    parser.add_argument("-d", "--depth", type=float, help="Max depth in cm")
    parser.add_argument("-m", "--min", type=float, default=0.1, help="Min depth in cm")
    parser.add_argument("-o", "--out", type=str, help="Path to output STL file")
    parser.add_argument("-i", "--invert", action="store_true", help="Invert grayscale")
    parser.add_argument("-s", "--step", type=int, default=10, help="Step size (1-100)")
    parser.add_argument("-g", "--gui", action="store_true", help="Launch Gradio GUI")

    args = parser.parse_args()

    if len(vars(args)) == 0 or args.gui:
        launch_gui()
        return

    if not (args.path and args.width and args.depth and args.out):
        print("Error: --path, --width, --depth, and --out are required unless using --gui")
        return

    img_array = load_image(args.path, args.invert)
    mesh = generate_mesh(img_array, args.width, args.depth, args.min, args.step)
    save_mesh(mesh, args.out)
    print(f"STL saved to {args.out}")


if __name__ == "__main__":
    main()
