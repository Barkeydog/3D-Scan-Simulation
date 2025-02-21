#!/usr/bin/env python
import numpy as np
import trimesh
import csv
import time

# ---------------------------
# Utility: Look-at transformation
# ---------------------------
def look_at(camera_position, target, up=np.array([0, 1, 0])):
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
# Generate camera rays using a pinhole camera model
# ---------------------------
def compute_camera_rays(camera_position, target, resolution, fov_deg=45.0):
    width, height = resolution
    fov = np.deg2rad(fov_deg)
    aspect_ratio = width / height
    image_plane_height = 2 * np.tan(fov / 2)
    image_plane_width = image_plane_height * aspect_ratio
    
    xs = np.linspace(-image_plane_width / 2, image_plane_width / 2, width)
    ys = np.linspace(-image_plane_height / 2, image_plane_height / 2, height)
    xv, yv = np.meshgrid(xs, ys)
    
    # Assume image plane at z = 1 in camera space.
    dirs_cam = np.stack([xv, yv, np.ones_like(xv)], axis=-1)
    dirs_cam = dirs_cam.reshape(-1, 3)
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=1, keepdims=True)
    
    # Transform ray directions to world coordinates.
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
    grid_size = 1.0  # total extent of the grid (world units)
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
    origins, directions = compute_camera_rays(camera_position, target, resolution, fov_deg)
    try:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
        locations, _, _ = intersector.intersects_location(
            ray_origins=origins, ray_directions=directions, multiple_hits=False)
    except Exception:
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=origins, ray_directions=directions, multiple_hits=False)
    return locations

# ---------------------------
# Determine if a mesh sample point is "scanned" by at least two cameras.
# ---------------------------
def is_sample_scanned(sample, camera_obs, camera_positions, baseline_threshold=0.1, eps=0.01):
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
# Simulation without visualization.
# ---------------------------
def simulate_scan_no_vis(mesh, num_rows, num_cols, resolution, fov_deg=45.0,
                         orbit_radius=3.0, num_surface_samples=1000,
                         baseline_threshold=0.1, eps=0.01):
    target = np.array([0.0, 0.0, 0.0])
    camera_positions = generate_camera_positions(num_rows, num_cols, orbit_radius)
    camera_obs = {}
    
    # For each camera, cast rays and store the intersections.
    for i, cam_pos in enumerate(camera_positions):
        pts = cast_camera_rays(mesh, cam_pos, target, resolution, fov_deg)
        camera_obs[i] = pts
    
    # Sample points on the mesh surface.
    samples, _ = trimesh.sample.sample_surface(mesh, num_surface_samples)
    scanned = 0
    for s in samples:
        if is_sample_scanned(s, camera_obs, camera_positions, baseline_threshold, eps):
            scanned += 1
    coverage = scanned / len(samples) * 100
    return coverage

# ---------------------------
# Main iteration routine.
# ---------------------------
def main():
    # Create a default mesh (icosphere)
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    
    # Define iteration parameters
    grid_sizes = [(2, 2), (3, 3), (4, 4)]  # (rows, cols)
    resolutions = [(64, 48), (128, 96)]      # (width, height)
    fov_values = [30.0, 45.0, 60.0]           # in degrees
    
    # Other simulation parameters
    num_surface_samples = 1000
    orbit_radius = 3.0
    baseline_threshold = 0.1
    eps = 0.01

    # Create a list to hold our results.
    # The first row is the header.
    results = []
    results.append(["Rows", "Cols", "Resolution", "FOV (deg)", "Coverage (%)", "Time (s)"])
    
    # Iterate through all combinations.
    for grid in grid_sizes:
        rows, cols = grid
        for resolution in resolutions:
            for fov in fov_values:
                start_time = time.time()
                coverage = simulate_scan_no_vis(
                    mesh,
                    num_rows=rows,
                    num_cols=cols,
                    resolution=resolution,
                    fov_deg=fov,
                    orbit_radius=orbit_radius,
                    num_surface_samples=num_surface_samples,
                    baseline_threshold=baseline_threshold,
                    eps=eps
                )
                elapsed = time.time() - start_time
                results.append([rows, cols, f"{resolution[0]}x{resolution[1]}", fov, f"{coverage:.2f}", f"{elapsed:.2f}"])
                print(f"Rows: {rows}, Cols: {cols}, Resolution: {resolution}, FOV: {fov} -> Coverage: {coverage:.2f}% in {elapsed:.2f}s")
    
    # Write the results to a CSV file.
    csv_filename = "iteration_results.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)
    
    print(f"\nIteration complete. Results saved to '{csv_filename}'.")

if __name__ == "__main__":
    main()
