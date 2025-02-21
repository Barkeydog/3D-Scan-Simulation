import numpy as np
import trimesh
import argparse
import time
import sys

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------------------------
# Utility: Look-at transformation
# ---------------------------
def look_at(camera_position, target, up=np.array([0, 1, 0])):
    """
    Compute a camera-to-world transformation matrix given the camera position,
    a target point, and an up vector.
    """
    forward = target - camera_position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    true_up = np.cross(forward, right)
    
    transform = np.eye(4)
    transform[:3, 0] = right
    transform[:3, 1] = true_up
    transform[:3, 2] = forward
    transform[:3, 3] = camera_position
    return transform

# ---------------------------
# Generate camera rays from a pinhole camera model
# ---------------------------
def compute_camera_rays(camera_position, target, resolution, fov_deg=45.0):
    """
    Given a camera position and target, compute (vectorized) rays from the camera
    using a pinhole model.
    
    Returns:
      origins: (N,3) array (all equal to camera_position)
      directions: (N,3) array (normalized ray directions in world coordinates)
    """
    width, height = resolution
    fov = np.deg2rad(fov_deg)
    aspect_ratio = width / height
    # Image plane dimensions (at z=1 in camera space)
    image_plane_height = 2 * np.tan(fov / 2)
    image_plane_width = image_plane_height * aspect_ratio

    xs = np.linspace(-image_plane_width / 2, image_plane_width / 2, width)
    ys = np.linspace(-image_plane_height / 2, image_plane_height / 2, height)
    xv, yv = np.meshgrid(xs, ys)
    
    # In camera space, assume image plane is at z = 1.
    dirs_cam = np.stack([xv, yv, np.ones_like(xv)], axis=-1)
    dirs_cam = dirs_cam.reshape(-1, 3)
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=1, keepdims=True)
    
    # Transform directions to world space.
    c2w = look_at(camera_position, target)
    R = c2w[:3, :3]
    directions = (R @ dirs_cam.T).T
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    origins = np.repeat(camera_position.reshape(1, 3), directions.shape[0], axis=0)
    return origins, directions

# ---------------------------
# Arrange cameras in a grid around the model.
# ---------------------------
def generate_camera_positions(num_rows, num_cols, orbit_radius=3.0):
    """
    Arrange cameras on a plane at x = orbit_radius with varying y and z.
    """
    grid_size = 1.0  # extent of the grid in world units
    ys = np.linspace(-grid_size/2, grid_size/2, num_rows)
    zs = np.linspace(-grid_size/2, grid_size/2, num_cols)
    positions = []
    for y in ys:
        for z in zs:
            pos = np.array([orbit_radius, y, z])
            positions.append(pos)
    return np.array(positions)

# ---------------------------
# Cast rays from one camera and return intersection points.
# ---------------------------
def cast_camera_rays(mesh, camera_position, target, resolution, fov_deg=45.0):
    """
    Cast rays from one camera using a pinhole model and return intersection points.
    """
    origins, directions = compute_camera_rays(camera_position, target, resolution, fov_deg)
    try:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
        locations, index_ray, index_tri = intersector.intersects_location(
            ray_origins=origins, ray_directions=directions, multiple_hits=False)
    except Exception:
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=origins, ray_directions=directions, multiple_hits=False)
    return locations

# ---------------------------
# Check if a sample point is “scanned” (i.e. seen by at least two cameras with sufficient baseline).
# ---------------------------
def is_sample_scanned(sample, camera_obs, camera_positions, baseline_threshold=0.1, eps=0.01):
    """
    Determine if sample is observed (within eps) by at least two cameras whose centers are
    separated by more than baseline_threshold.
    """
    observed_cams = []
    for cam_idx, pts in camera_obs.items():
        if pts.shape[0] == 0:
            continue
        dists = np.linalg.norm(pts - sample, axis=1)
        if np.any(dists < eps):
            observed_cams.append(cam_idx)
    if len(observed_cams) < 2:
        return False
    for i in range(len(observed_cams)):
        for j in range(i+1, len(observed_cams)):
            p1 = camera_positions[observed_cams[i]]
            p2 = camera_positions[observed_cams[j]]
            if np.linalg.norm(p1 - p2) > baseline_threshold:
                return True
    return False

# ---------------------------
# Visualization and scanning simulation.
# ---------------------------
def visualize_scan(mesh, num_rows, num_cols, resolution,
                   fov_deg=45.0, orbit_radius=3.0,
                   num_surface_samples=1000,
                   baseline_threshold=0.1, eps=0.01):
    """
    Load the mesh and arrange cameras. Then for each camera, cast rays and update
    a live 3D plot to show the intersection points. Finally, sample the surface
    to determine which parts of the mesh are scanned (seen by at least two cameras
    with a sufficient baseline) and update the plot.
    """
    target = np.array([0.0, 0.0, 0.0])
    camera_positions = generate_camera_positions(num_rows, num_cols, orbit_radius)
    num_cameras = len(camera_positions)
    camera_obs = {}  # will store intersection points per camera

    # Set up Matplotlib 3D figure.
    plt.ion()  # interactive mode on
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Scanning Simulation")

    # Plot the mesh using a Poly3DCollection.
    # Get the vertices for each face.
    mesh_triangles = mesh.vertices[mesh.faces]
    mesh_collection = Poly3DCollection(mesh_triangles, alpha=0.2, facecolor='cyan', edgecolor='gray')
    ax.add_collection3d(mesh_collection)

    # Set the axis limits based on the mesh bounds.
    bounds = mesh.bounds
    pad = 1.0
    ax.set_xlim(bounds[0, 0]-pad, bounds[1, 0]+pad)
    ax.set_ylim(bounds[0, 1]-pad, bounds[1, 1]+pad)
    ax.set_zlim(bounds[0, 2]-pad, bounds[1, 2]+pad)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Plot camera positions.
    ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
               c='blue', marker='^', s=100, label='Cameras')
    plt.legend()

    plt.draw()
    plt.pause(1.0)

    # Process each camera: cast rays and update plot with intersection points.
    for i, cam_pos in enumerate(camera_positions):
        pts = cast_camera_rays(mesh, cam_pos, target, resolution, fov_deg)
        camera_obs[i] = pts
        if pts.shape[0] > 0:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                       c='red', s=2, alpha=0.5, label=f'Camera {i} Intersections' if i==0 else None)
        ax.text(cam_pos[0], cam_pos[1], cam_pos[2], f'{i}', color='black')
        plt.draw()
        plt.pause(0.5)  # pause to update visualization

    # Now, sample points on the mesh surface.
    samples, _ = trimesh.sample.sample_surface(mesh, num_surface_samples)
    scanned = 0
    scanned_points = []
    for s in samples:
        if is_sample_scanned(s, camera_obs, camera_positions, baseline_threshold, eps):
            scanned += 1
            scanned_points.append(s)
    coverage = scanned / len(samples) * 100

    scanned_points = np.array(scanned_points)
    if scanned_points.shape[0] > 0:
        ax.scatter(scanned_points[:, 0], scanned_points[:, 1], scanned_points[:, 2],
                   c='green', s=5, label='Scanned Surface Points')
        plt.legend()
    plt.draw()
    plt.ioff()  # turn interactive mode off
    plt.show()

    print(f"\nTotal scanned surface: {coverage:.2f}% (based on {num_surface_samples} samples)")
    return coverage

# ---------------------------
# Load a mesh from file or use a default.
# ---------------------------
def load_mesh(model_path=None):
    """
    If a model_path is provided, load that file.
    Otherwise, return a default icosphere.
    """
    if model_path is not None:
        try:
            mesh = trimesh.load(model_path)
            if not isinstance(mesh, trimesh.Trimesh):
                # In case the file contains multiple meshes, combine them.
                mesh = trimesh.util.concatenate(mesh.dump())
            print(f"Loaded mesh from {model_path}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            return mesh
        except Exception as e:
            print(f"Error loading mesh from {model_path}: {e}")
            sys.exit(1)
    else:
        print("No model file provided. Using default icosphere.")
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        return mesh

# ---------------------------
# Main routine: parse arguments, load model, and run visualization.
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="3D Scanning Simulation with Visualization")
    parser.add_argument('--model', type=str, default=None,
                        help="Path to your 3D model file (e.g., OBJ, STL, PLY). If omitted, a default icosphere is used.")
    parser.add_argument('--rows', type=int, default=2, help="Number of camera rows in the grid")
    parser.add_argument('--cols', type=int, default=2, help="Number of camera columns in the grid")
    parser.add_argument('--width', type=int, default=64, help="Camera resolution width (pixels)")
    parser.add_argument('--height', type=int, default=48, help="Camera resolution height (pixels)")
    parser.add_argument('--fov', type=float, default=45.0, help="Camera field of view in degrees")
    parser.add_argument('--samples', type=int, default=1000, help="Number of surface samples to estimate coverage")
    args = parser.parse_args()

    resolution = (args.width, args.height)
    mesh = load_mesh(args.model)
    
    # Run the simulation with live visualization.
    coverage = visualize_scan(mesh, args.rows, args.cols, resolution,
                              fov_deg=args.fov, orbit_radius=3.0,
                              num_surface_samples=args.samples,
                              baseline_threshold=0.1, eps=0.01)

if __name__ == '__main__':
    main()
