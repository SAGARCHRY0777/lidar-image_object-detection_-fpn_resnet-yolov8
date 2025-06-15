# bev_projection_utils.py
import numpy as np
from typing import List, Tuple, Dict, Any

def get_3d_box_corners(center: np.ndarray, dims: np.ndarray, heading: float) -> np.ndarray:
    """
    Calculates the 8 corners of a 3D bounding box.

    Args:
        center: A 3-element numpy array (x, y, z) representing the box center in ego-vehicle frame.
        dims: A 3-element numpy array (length, width, height). Length along x, width along y, height along z.
        heading: Yaw angle in radians, representing rotation around the z-axis.

    Returns:
        A (8, 3) numpy array where each row is an (x, y, z) corner coordinate.
    """
    length, width, height = dims[0], dims[1], dims[2]

    # Create local coordinates for the 8 corners
    x_corners = [length / 2, length / 2, -length / 2, -length / 2, length / 2, length / 2, -length / 2, -length / 2]
    y_corners = [width / 2, -width / 2, -width / 2, width / 2, width / 2, -width / 2, -width / 2, width / 2]
    z_corners = [height / 2, height / 2, height / 2, height / 2, -height / 2, -height / 2, -height / 2, -height / 2]

    corners = np.array([x_corners, y_corners, z_corners])

    # Apply rotation around Z-axis (yaw)
    rotation_matrix = np.array([
        [np.cos(heading), -np.sin(heading), 0],
        [np.sin(heading), np.cos(heading), 0],
        [0, 0, 1]
    ])
    rotated_corners = rotation_matrix @ corners

    # Translate to the global center
    translated_corners = rotated_corners + center.reshape(3, 1)

    return translated_corners.T # Transpose to get (8, 3)


def project_box_to_bev(box_corners_3d_ego: np.ndarray) -> np.ndarray:
    """
    Projects 3D bounding box corners (in ego-vehicle frame) onto the BEV plane.
    The BEV plane is typically the ground plane (Z=0) in the ego-vehicle coordinate system.

    Args:
        box_corners_3d_ego: A (8, 3) numpy array of 3D box corners in the ego-vehicle frame.

    Returns:
        A (8, 2) numpy array of (x, y) coordinates projected onto the BEV plane.
    """
    # Simply take the X and Y coordinates, effectively projecting onto the Z=0 plane
    return box_corners_3d_ego[:, :2]


def draw_bev_boxes(
    ax,
    box_corners_bev: np.ndarray,
    color: str = 'red',
    linewidth: float = 2.0,
    alpha: float = 0.7
):
    """
    Draws 2D bounding boxes on a matplotlib BEV plot.

    Args:
        ax: Matplotlib axes object.
        box_corners_bev: A (N, 8, 2) numpy array, where N is the number of boxes,
                         and each (8, 2) array represents the 2D corners of a box.
        color: Color for the bounding boxes.
        linewidth: Line width for the box edges.
        alpha: Transparency of the box.
    """
    for box in box_corners_bev:
        # Define the order of corners to draw the rectangle
        # Assuming corners are ordered: front-right-top, front-left-top, rear-left-top, rear-right-top
        # For BEV, we just need the 4 bottom corners, or consistently trace the top ones if they are above ground.
        # Let's assume the box_corners_bev gives us the 8 corners, and we want to draw the base.
        # A common way to get the base for a standard box corner order is:
        # 0: (x_max, y_max), 1: (x_max, y_min), 2: (x_min, y_min), 3: (x_min, y_max)
        # Assuming the first 4 corners define the top and the next 4 define the bottom
        # or we just want the convex hull on the XY plane.
        # For simplicity, let's connect them in a way that forms a rectangle on the BEV plane.
        # We need to ensure the corners are ordered correctly for drawing a polygon.
        # Given the get_3d_box_corners output, let's find the min/max x, y to define the rectangle outline.
        # This is a simplification; for oriented boxes, you connect the actual projected corners.

        # To properly draw an oriented rectangle, you need to know which corners form the base.
        # Based on get_3d_box_corners, indices might be:
        # 4, 5, 6, 7 are typically the bottom corners.
        # Let's connect 4->5->6->7->4
        # Or, we can use the original 3D box logic for connectivity:
        # Edge connections (simplified for BEV):
        # We need to connect the four corners that define the base rectangle.
        # Let's assume the output from get_3d_box_corners, if projected to Z=0,
        # will yield 4 distinct (x,y) points that define the base polygon.
        # We can then connect them. A simple convex hull might work for drawing.

        # More robust approach: connect specific corners to form the base rectangle.
        # Assuming get_3d_box_corners produces corners in a consistent order:
        # For a box at (0,0,0) with positive dims, (x,y) corners might be:
        # (-L/2, -W/2), (L/2, -W/2), (L/2, W/2), (-L/2, W/2) for bottom
        # Let's consider `box` as the (8, 2) array of projected points.
        # We need to pick 4 corners that form the base rectangle in BEV.
        # A typical order (after rotation and translation) for a base could be:
        # front-right-bottom, front-left-bottom, rear-left-bottom, rear-right-bottom
        # This requires understanding the exact output order of `get_3d_box_corners`.

        # A robust way is to find the convex hull of the 2D projected points, if the box isn't axis-aligned.
        # However, for 3D boxes projected to 2D, the base rectangle is simply four of the 8 projected points.
        # Based on the typical order of get_3d_box_corners (which goes z_corners = [height / 2, ..., -height / 2, ...]):
        # corners[4], corners[5], corners[6], corners[7] correspond to the bottom corners.
        # So we can connect these 4 points.
        
        # Ensure the box has at least 4 corners to draw a polygon
        if box.shape[0] >= 4:
            # Taking the first 4 for simplicity, or specify indices if known
            # A more robust way to draw the base for an arbitrary oriented box is
            # to consider the bottom 4 points of the 3D box, project them,
            # and then connect them in order.
            # From get_3d_box_corners, indices 4,5,6,7 correspond to z=-height/2
            # These are: [L/2, W/2, -H/2], [L/2, -W/2, -H/2], [-L/2, -W/2, -H/2], [-L/2, W/2, -H/2]
            # When projected to BEV, they become (L/2, W/2), (L/2, -W/2), (-L/2, -W/2), (-L/2, W/2) after rotation.
            # So, draw these four points in order:
            # We connect projected_corners[4] -> projected_corners[5] -> projected_corners[6] -> projected_corners[7] -> projected_corners[4]
            # This order corresponds to the bottom rectangle.

            # Example: Connect the bottom 4 corners (indices 4, 5, 6, 7)
            # Make sure these indices are consistent with how get_3d_box_corners generates points.
            # If `box` is the (8,2) projected corners, then use those indices.
            
            # This assumes that box_corners_bev already has the 8 projected points.
            # You need to manually pick the 4 corners that form the base and connect them.
            # A common way to draw an oriented bounding box in 2D is to draw the 4 edges of the base.
            # For a 3D box, the 4 base points are typically generated at the bottom.
            # Assuming get_3d_box_corners returns in a consistent order, e.g.,
            # 0,1,2,3 for top, 4,5,6,7 for bottom.
            # We can draw the lines: (4,5), (5,6), (6,7), (7,4)
            # And also potentially draw a line from front-center to indicate heading.

            # Here's a generic way to draw a polygon for a 4-point rectangle:
            # We assume the input `box` (8,2) contains the 8 projected 2D points.
            # We need to extract the 4 unique base points.
            # Or, if `box_corners_bev` was already processed to only contain 4 corners for the base:
            # For now, let's assume `box` is (4,2) representing the base corners in order.
            # If `box` is (8,2), we should select the bottom 4 unique points and connect them.
            
            # Let's refine based on get_3d_box_corners output indices:
            # Indices 4, 5, 6, 7 are the bottom corners in 3D. When projected, these become the 2D base.
            base_corners = box[[4, 5, 6, 7]]
            
            # Connect the corners
            ax.plot([base_corners[0, 0], base_corners[1, 0]], [base_corners[0, 1], base_corners[1, 1]], color=color, linewidth=linewidth, alpha=alpha)
            ax.plot([base_corners[1, 0], base_corners[2, 0]], [base_corners[1, 1], base_corners[2, 1]], color=color, linewidth=linewidth, alpha=alpha)
            ax.plot([base_corners[2, 0], base_corners[3, 0]], [base_corners[2, 1], base_corners[3, 1]], color=color, linewidth=linewidth, alpha=alpha)
            ax.plot([base_corners[3, 0], base_corners[0, 0]], [base_corners[3, 1], base_corners[0, 1]], color=color, linewidth=linewidth, alpha=alpha)

            # Optional: draw a line indicating the front of the box
            # Assuming the front is along the positive X axis of the box.
            # The mid-point of the front edge can be calculated.
            # For corners 0 and 1 (top front), or 4 and 5 (bottom front):
            # Using bottom front corners for BEV:
            front_center_bottom_x = (base_corners[0, 0] + base_corners[1, 0]) / 2
            front_center_bottom_y = (base_corners[0, 1] + base_corners[1, 1]) / 2
            
            # Point slightly ahead of front center to indicate heading
            # This requires knowing the heading of the box relative to BEV coordinate system.
            # If the box is oriented by `heading`, then the front is along its local x-axis.
            # We can calculate a small offset in that direction from the center of the front edge.
            
            # Re-calculate center from base corners for this purpose
            box_center_bev_x = np.mean(base_corners[:, 0])
            box_center_bev_y = np.mean(base_corners[:, 1])

            # Small line indicating heading. Length 'indicator_length'.
            indicator_length = 1.0 # Adjust as needed
            
            # From the original heading, calculate the direction vector
            direction_x = np.cos(np.radians(box_corners_bev[0,0])) # This is wrong, heading is not in box_corners_bev directly.
                                                                # You need to pass the original heading for the box.
            
            # For demonstration, let's simplify and just draw the base.
            # If you want to indicate heading, you'll need the original box heading.
            # For now, just the base rectangle is drawn.
            pass


def example_usage_bev_projection(
    calib_filepath: str,
    lidar_points: np.ndarray, # N, 4 (x,y,z,intensity)
    box_data: List[Dict[str, Any]] # List of dicts, each with 'center', 'dims', 'heading'
):
    """
    Demonstrates how to project LiDAR points and bounding boxes onto a BEV map.
    
    Args:
        calib_filepath: Path to the calibration JSON file.
        lidar_points: Nx4 numpy array of LiDAR points (x, y, z, intensity) in LiDAR frame.
        box_data: List of dictionaries, each containing:
                  'center': [x, y, z] (list or np.ndarray) in ego-vehicle frame
                  'dims': [length, width, height] (list or np.ndarray)
                  'heading': yaw angle in radians
    """
    # For BEV, we generally don't need camera calibration directly,
    # but we do need the LiDAR to ego-vehicle transform from the calibration file.
    import json
    with open(calib_filepath, "r") as f:
        calib = json.load(f)

    # Get LiDAR to Ego-vehicle transform
    lidar_data_raw = calib['lidar_data'][0]['value'] # Assuming 'down_lidar' is the first entry
    rot_coeffs_lv = lidar_data_raw['vehicle_SE3_down_lidar_']['rotation']['coefficients'] # (qx, qy, qz, qw)
    trans_coeffs_lv = np.array(lidar_data_raw['vehicle_SE3_down_lidar_']['translation'], dtype=np.float32)
    
    from new_se3 import SE3
    from new_transform_utils import quat2rotmat
    R_l_v = quat2rotmat(np.array([rot_coeffs_lv[3], rot_coeffs_lv[0], rot_coeffs_lv[1], rot_coeffs_lv[2]]))
    T_l_v_matrix = SE3(rotation=R_l_v, translation=trans_coeffs_lv).transform_matrix # Transform from LiDAR to Ego-vehicle

    # Project LiDAR points to Ego-vehicle frame
    lidar_points_h = np.hstack([lidar_points[:, :3], np.ones((lidar_points.shape[0], 1))]) # Add homogeneous coord
    lidar_points_ego_hom = T_l_v_matrix @ lidar_points_h.T # (4, N)
    lidar_points_ego = lidar_points_ego_hom[:3, :].T # (N, 3)

    # Filter LiDAR points for ground plane (e.g., z within a range for visualization)
    # This is often needed for a clean BEV display
    ground_z_min = -2.0 # Adjust as per your ground plane definition
    ground_z_max = 0.5
    valid_lidar_indices = (lidar_points_ego[:, 2] > ground_z_min) & (lidar_points_ego[:, 2] < ground_z_max)
    lidar_points_on_ground = lidar_points_ego[valid_lidar_indices]

    # Prepare bounding box corners for BEV projection
    all_projected_boxes_bev = []
    for box in box_data:
        center_ego = np.array(box['center'], dtype=np.float32)
        dims = np.array(box['dims'], dtype=np.float32)
        heading = box['heading']

        box_corners_3d_ego = get_3d_box_corners(center_ego, dims, heading)
        projected_box_bev = project_box_to_bev(box_corners_3d_ego)
        all_projected_boxes_bev.append(projected_box_bev)

    # Visualization (using matplotlib)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect('equal', adjustable='box')

    # Plot LiDAR points on BEV
    ax.scatter(lidar_points_on_ground[:, 0], lidar_points_on_ground[:, 1], s=1, alpha=0.5, color='gray')

    # Plot bounding boxes on BEV
    for projected_box_bev in all_projected_boxes_bev:
        draw_bev_boxes(ax, np.array([projected_box_bev])) # Pass as (N, 8, 2)
    
    # Set plot limits (adjust based on your scene size)
    ax.set_xlim([-50, 50])
    ax.set_ylim([-50, 50])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Bird's Eye View with LiDAR Points and Bounding Boxes")
    ax.grid(True)
    plt.show()

# You would call example_usage_bev_projection with your actual data
# For example:
# from bev_projection_utils import example_usage_bev_projection
# dummy_lidar_points = np.random.rand(1000, 4) * 100 - 50 # Example: 1000 random points
# dummy_box_data = [
#     {'center': [5.0, 2.0, -1.0], 'dims': [4.0, 1.8, 1.5], 'heading': np.radians(30)},
#     {'center': [-10.0, -5.0, -1.0], 'dims': [3.0, 1.5, 1.6], 'heading': np.radians(-60)}
# ]
# example_usage_bev_projection('path/to/your/calibration.json', dummy_lidar_points, dummy_box_data)