import numpy as np
import json
import os
import cv2
from typing import NamedTuple, Dict, Any, Tuple, List # For type hinting

# Assume this is relative to your SFA root. Adjust if needed.
import config.argoverse_config as cnf

# Import the corrected quaternion to rotation matrix utility
from new_transform_utils import quat2rotmat

# --- Start: Replicated Calibration and Transformation Helpers (No Argoverse API) ---

class CameraConfig(NamedTuple):
    """Camera config for extrinsic matrix, intrinsic matrix, image width/height.
    Args:
        extrinsic: extrinsic matrix (camera_SE3_egovehicle)
        intrinsic: intrinsic matrix
        img_width: image width
        img_height: image height
        distortion_coeffs: distortion coefficients
    """
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    img_width: int
    img_height: int
    distortion_coeffs: np.ndarray


def quat_coeffs_to_rotation_matrix(coeffs: List[float]) -> np.ndarray:
    """
    Convert Argoverse quaternion coefficients (qx, qy, qz, qw) to a 3x3 rotation matrix
    using the new_transform_utils.quat2rotmat function.
    """
    # new_transform_utils.quat2rotmat expects (w, x, y, z)
    # Argoverse coeffs are (qx, qy, qz, qw)
    return quat2rotmat(np.array([coeffs[3], coeffs[0], coeffs[1], coeffs[2]]))


def build_se3_transform(rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
    """
    Builds a 4x4 SE3 transformation matrix from a 3x3 rotation matrix and a 3-element translation vector.
    """
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation_vector.flatten()
    return T


def get_camera_extrinsic_matrix_no_api(config_data: Dict[str, Any]) -> np.ndarray:
    """
    Load camera calibration rotation and translation to form the extrinsic matrix (camera_SE3_egovehicle).
    This matrix transforms points from the ego-vehicle frame to the camera frame.
    The input `config_data` is the 'value' dictionary for a specific camera from the calibration JSON.
    """
    vehicle_SE3_sensor = config_data["vehicle_SE3_camera_"]
    egovehicle_t_camera = np.array(vehicle_SE3_sensor["translation"], dtype=np.float32)
    egovehicle_q_camera = vehicle_SE3_sensor["rotation"]["coefficients"] # (qx, qy, qz, qw)
    
    # Use the corrected quat_coeffs_to_rotation_matrix
    R_camera_egovehicle = quat_coeffs_to_rotation_matrix(egovehicle_q_camera)
    
    # Build the SE3 transform
    T_camera_egovehicle = build_se3_transform(R_camera_egovehicle, egovehicle_t_camera)
    return T_camera_egovehicle


def get_camera_intrinsic_matrix_no_api(camera_config_data: Dict[str, Any]) -> np.ndarray:
    """
    Load camera intrinsic matrix.
    The input `camera_config_data` is the 'value' dictionary for a specific camera.
    """
    intrinsic_matrix = np.eye(3, dtype=np.float32)
    intrinsic_matrix[0, 0] = camera_config_data["focal_length_x_px_"]
    intrinsic_matrix[0, 1] = camera_config_data["skew_"]
    intrinsic_matrix[0, 2] = camera_config_data["focal_center_x_px_"]
    intrinsic_matrix[1, 1] = camera_config_data["focal_length_y_px_"]
    intrinsic_matrix[1, 2] = camera_config_data["focal_center_y_px_"]
    intrinsic_matrix[2, 2] = 1.0
    return intrinsic_matrix


def get_image_dims_for_camera_no_api(camera_name: str) -> Tuple[int, int]:
    """
    Provides image dimensions (width, height) for specified Argoverse cameras.
    These values are hardcoded based on typical Argoverse camera resolutions.
    """
    if "front" in camera_name:
        return 1920, 1200 # Example: Full HD width, 1200 height
    elif "side" in camera_name or "rear" in camera_name:
        return 1280, 960 # Example: Common resolution for side/rear cameras
    else:
        raise ValueError(f"Unknown camera name for dimensions: {camera_name}")


class ArgoverseCalibration:
    """
    Handles Argoverse camera and LiDAR calibration.
    Simplified, not using Argoverse API directly.
    """
    def __init__(self, calib_filepath: str, target_camera: str = 'image_raw_ring_front_center'):
        self.calib_filepath = calib_filepath
        self.target_camera = target_camera
        self.calib_data = self._load_calibration_data()
        self.camera_config = self._get_camera_config()
        self.L2C = self._get_lidar_to_camera_transform() # LiDAR to Camera (Ego-vehicle to Camera)
        self.P2 = self.camera_config.intrinsic # Camera intrinsic matrix

    def _load_calibration_data(self) -> Dict[str, Any]:
        """Loads calibration data from the JSON file."""
        with open(self.calib_filepath, 'r') as f:
            calib = json.load(f)
        return calib

    def _get_camera_config(self) -> CameraConfig:
        """Extracts and returns CameraConfig for the target camera."""
        camera_data = None
        for cam in self.calib_data['camera_data']:
            if cam['key'] == self.target_camera:
                camera_data = cam['value']
                break
        if camera_data is None:
            raise ValueError(f"Calibration data for camera {self.target_camera} not found.")

        extrinsic_matrix = get_camera_extrinsic_matrix_no_api(camera_data)
        intrinsic_matrix = get_camera_intrinsic_matrix_no_api(camera_data)
        img_width, img_height = get_image_dims_for_camera_no_api(self.target_camera)
        print("#################################################################")
        print("Camera calibration data loaded:")
        print("extrinsic_matrix, intrinsic_matrix, img_width, img_height=",extrinsic_matrix, intrinsic_matrix, img_width, img_height)
        print("#################################################################")
        # Distortion coefficients are often present in calibration JSON.
        # Assuming `camera_data["distortion_coeffs"]` exists.
        distortion_coeffs = np.array(camera_data.get("distortion_coeffs", [-0.16983475865148748, 0.1189081299929571, -0.02488434834889849]), dtype=np.float32)

        return CameraConfig(extrinsic_matrix, intrinsic_matrix, img_width, img_height, distortion_coeffs)

    def _get_lidar_to_camera_transform(self) -> np.ndarray:
        """
        Computes the transformation matrix from LiDAR frame to the target camera frame.
        This involves: LiDAR -> Ego-vehicle -> Camera
        Assumes the LiDAR frame is the 'down_lidar' and camera frame is 'ring_front_center' or similar.
        """
        # LiDAR to Ego-vehicle transformation (vehicle_SE3_down_lidar_)
        lidar_data_raw = self.calib_data['lidar_data'][0]['value'] # Assuming 'down_lidar' is the first entry
        rot_coeffs_lv = lidar_data_raw['vehicle_SE3_down_lidar_']['rotation']['coefficients'] # (qx, qy, qz, qw)
        trans_coeffs_lv = np.array(lidar_data_raw['vehicle_SE3_down_lidar_']['translation'], dtype=np.float32)
        
        R_l_v = quat2rotmat(np.array([rot_coeffs_lv[3], rot_coeffs_lv[0], rot_coeffs_lv[1], rot_coeffs_lv[2]])) # Reorder
        T_l_v = build_se3_transform(R_l_v, trans_coeffs_lv) # T_ego_lidar

        # Camera to Ego-vehicle transformation (vehicle_SE3_camera_)
        # This is the inverse of camera_SE3_egovehicle (extrinsic matrix).
        # We already have camera_SE3_egovehicle as self.camera_config.extrinsic
        # So, T_v_c = (T_c_v)^-1
        T_c_v_inv = np.linalg.inv(self.camera_config.extrinsic)

        # LiDAR to Camera: T_c_l = T_c_v @ T_v_l
        # T_v_l is the transform from LiDAR to vehicle.
        # So L2C = T_c_v @ T_v_l = T_c_v @ T_l_v (if T_l_v is vehicle_SE3_lidar)
        # However, `vehicle_SE3_down_lidar_` is `vehicle_from_lidar`, so T_l_v.
        # So L2C = T_c_v @ T_l_v
        # T_c_v is inv(extrinsic)
        L2C_matrix = T_c_v_inv @ T_l_v
        return L2C_matrix

    def project_lidar_to_camera_image(self, lidar_points: np.ndarray) -> np.ndarray:
        """
        Projects 3D LiDAR points (N, 3) from the LiDAR frame to 2D image coordinates (N, 2).
        This involves: LiDAR -> Camera -> Image plane.
        """
        # Convert LiDAR points to homogeneous coordinates (N, 4)
        points_hom = np.hstack([lidar_points, np.ones((lidar_points.shape[0], 1))]).T # (4, N)

        # Transform points from LiDAR frame to Camera frame using L2C matrix (4, N)
        # L2C is camera_SE3_lidar
        points_cam_hom = self.L2C @ points_hom

        # Convert to 3D camera coordinates (3, N)
        points_cam = points_cam_hom[:3, :]

        # Apply intrinsic matrix to project to 2D image coordinates (3, N)
        uv_hom = self.P2 @ points_cam # (3, N)

        # Normalize by depth (Z-coordinate)
        depth = uv_hom[2, :]
        # Handle division by zero for points behind the camera or at infinity
        valid_indices = depth > 1e-6 # A small positive threshold
        
        uv_norm = np.full(uv_hom.shape, np.nan)
        if np.any(valid_indices):
            uv_norm[:, valid_indices] = uv_hom[:, valid_indices] / depth[valid_indices]

        # Extract u, v coordinates (2, N)
        uv = uv_norm[:2, :].T # (N, 2)

        return uv, valid_indices


    def project_ego_to_image(self, points_ego: np.ndarray) -> np.ndarray:
        """
        Projects 3D points from ego-vehicle frame to 2D image coordinates.
        This uses the camera's extrinsic and intrinsic matrices.
        points_ego: Nx3 array of points in ego-vehicle frame.
        """
        # Convert ego-vehicle points to homogeneous coordinates (N, 4)
        points_hom = np.hstack([points_ego, np.ones((points_ego.shape[0], 1))]).T # (4, N)

        # Transform points from ego-vehicle frame to Camera frame using extrinsic matrix (4, N)
        # self.camera_config.extrinsic is camera_SE3_egovehicle
        points_cam_hom = self.camera_config.extrinsic @ points_hom

        # Convert to 3D camera coordinates (3, N)
        points_cam = points_cam_hom[:3, :]

        # Apply intrinsic matrix to project to 2D image coordinates (3, N)
        uv_hom = self.P2 @ points_cam # (3, N)

        # Normalize by depth (Z-coordinate)
        depth = uv_hom[2, :]
        valid_indices = depth > 1e-6 # Small positive threshold

        uv_norm = np.full(uv_hom.shape, np.nan)
        if np.any(valid_indices):
            uv_norm[:, valid_indices] = uv_hom[:, valid_indices] / depth[valid_indices]

        # Extract u, v coordinates (2, N)
        uv = uv_norm[:2, :].T # (N, 2)
        return uv, valid_indices


def get_filtered_lidar(lidar_data: np.ndarray, boundary: Dict[str, float]) -> np.ndarray:
    """
    Filters LiDAR points based on defined boundary.
    lidar_data: Nx4 array (x, y, z, intensity)
    boundary: dictionary with min/max X, Y, Z
    """
    x, y, z, i = lidar_data[:, 0], lidar_data[:, 1], lidar_data[:, 2], lidar_data[:, 3]

    mask = (x >= boundary["minX"]) & \
           (x <= boundary["maxX"]) & \
           (y >= boundary["minY"]) & \
           (y <= boundary["maxY"]) & \
           (z >= boundary["minZ"]) & \
           (z <= boundary["maxZ"])
    
    return lidar_data[mask]


def makeBEVMap(lidar_points: np.ndarray, boundary: Dict[str, float], discretization: float) -> np.ndarray:
    """
    Generates a Bird's Eye View (BEV) map from LiDAR points.
    lidar_points: Nx4 array (x, y, z, intensity) in ego-vehicle coordinates.
    boundary: dictionary with min/max X, Y, Z for the BEV map.
    discretization: meters per pixel.
    Returns: 3-channel BEV map (height, width, 3) where channels typically represent
             height, intensity, and density.
    """
    # Filter points outside the boundary
    x, y, z, i = lidar_points[:, 0], lidar_points[:, 1], lidar_points[:, 2], lidar_points[:, 3]

    mask = (x >= boundary["minX"]) & (x < boundary["maxX"]) & \
           (y >= boundary["minY"]) & (y < boundary["maxY"]) & \
           (z >= boundary["minZ"]) & (z < boundary["maxZ"])

    x, y, z, i = x[mask], y[mask], z[mask], i[mask]

    # Calculate BEV dimensions
    H = int((boundary["maxX"] - boundary["minX"]) / discretization)
    W = int((boundary["maxY"] - boundary["minY"]) / discretization)

    # Convert to grid indices (row, col)
    # x-coordinates map to rows (decreasing with increasing x in Argoverse BEV convention)
    # y-coordinates map to columns (increasing with increasing y in Argoverse BEV convention)
    x_img = np.clip(((boundary["maxX"] - x) / discretization).astype(np.int32), 0, H - 1)
    y_img = np.clip(((y - boundary["minY"]) / discretization).astype(np.int32), 0, W - 1)

    # Initialize feature maps
    height_map = np.zeros((H, W), dtype=np.float32)
    intensity_map = np.zeros((H, W), dtype=np.float32)
    density_map = np.zeros((H, W), dtype=np.float32)

    # Populate maps (simple max pooling for height, sum for density)
    # Iterate through points and update maps
    for i_idx, (r, c) in enumerate(zip(x_img, y_img)):
        height_map[r, c] = max(height_map[r, c], z[i_idx]) # Max height in cell
        intensity_map[r, c] = max(intensity_map[r, c], i[i_idx]) # Max intensity in cell
        density_map[r, c] += 1 # Count points in cell

    # Normalize density map for visualization
    density_map = np.log1p(density_map) # Apply log1p for better visual contrast
    density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Stack channels to create BEV map
    # Order: Height, Intensity, Density (or whatever is desired)
    # Ensure all channels are scaled to 0-255 if for visualization
    height_map = cv2.normalize(height_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    intensity_map = cv2.normalize(intensity_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    bev_map = np.stack([density_map, height_map, intensity_map], axis=-1) # BGR format if using OpenCV imshow
    return bev_map


def create_3d_bbox_corners(h: float, w: float, l: float,
                           cx: float, cy: float, cz: float, yaw: float) -> np.ndarray:
    """
    Creates 8 corners of a 3D bounding box given dimensions, center, and yaw.
    h, w, l: height, width, length of the box.
    cx, cy, cz: center coordinates (x, y, z) in ego-vehicle frame.
    yaw: rotation around Z-axis (in radians).
    Returns: (8, 3) numpy array of 3D corner coordinates.
    """
    # Create base corners for a box centered at origin, aligned with axes
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    
    corners = np.vstack([x_corners, y_corners, z_corners]) # (3, 8)

    # Rotation matrix around Z-axis
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # Rotate corners
    rotated_corners = R_z @ corners # (3, 8)

    # Translate corners to the object's center
    translated_corners = rotated_corners + np.array([[cx], [cy], [cz]]) # (3, 8)
    
    return translated_corners.T # (8, 3)


def draw_box_3d_on_bev(bev_map: np.ndarray, box_3d: np.ndarray, color: Tuple[int, int, int], thickness: int,
                       boundary: Dict[str, float], discretization: float) -> np.ndarray:
    """
    Draws a 3D bounding box on the BEV map.
    bev_map: HxWx3 BEV image.
    box_3d: (8, 3) array of 3D corners of the bounding box in ego-vehicle coordinates.
    color: BGR tuple.
    thickness: line thickness.
    boundary: BEV boundary limits.
    discretization: meters per pixel.
    """
    draw_map = bev_map.copy()

    # Project 3D corners to BEV map coordinates (pixel coordinates)
    # x_img = (boundary["maxX"] - x) / discretization
    # y_img = (y - boundary["minY"]) / discretization
    
    x_coords = ((boundary["maxX"] - box_3d[:, 0]) / discretization).astype(np.int32)
    y_coords = ((box_3d[:, 1] - boundary["minY"]) / discretization).astype(np.int32)

    # Create a list of 2D points for the polygon (bottom face of the box)
    # The order of corners might need adjustment to connect them correctly
    # Assuming the first 4 corners define the bottom face in a sequential order
    # (front-right, front-left, rear-left, rear-right relative to vehicle front)
    # You might need to reorder `box_3d` corners depending on how they are generated.
    
    # Simple approach: draw lines between expected connected corners for a cuboid
    # Define connections for the bottom face (indices 0-3) and top face (indices 4-7)
    # and vertical edges.
    
    # Example connections for a standard corner order (e.g., from create_3d_bbox_corners)
    # Corners are typically:
    # 0: front-right-top
    # 1: front-left-top
    # 2: rear-left-top
    # 3: rear-right-top
    # 4: front-right-bottom
    # 5: front-left-bottom
    # 6: rear-left-bottom
    # 7: rear-right-bottom

    # Indices for the bottom rectangle (assuming it's 4,5,6,7)
    bottom_face_indices = [4, 5, 6, 7, 4] # Connect 4-5-6-7-4
    
    # Check if projected points are within BEV map boundaries before drawing
    H, W, _ = bev_map.shape
    valid_x_coords = np.clip(x_coords, 0, H - 1)
    valid_y_coords = np.clip(y_coords, 0, W - 1)
    
    # Draw the bottom face
    for i in range(len(bottom_face_indices) - 1):
        pt1_idx = bottom_face_indices[i]
        pt2_idx = bottom_face_indices[i+1]
        pt1 = (valid_y_coords[pt1_idx], valid_x_coords[pt1_idx])
        pt2 = (valid_y_coords[pt2_idx], valid_x_coords[pt2_idx])
        cv2.line(draw_map, pt1, pt2, color, thickness, cv2.LINE_AA)

    return draw_map


def draw_box_2d_on_image(img: np.ndarray, box_2d: np.ndarray, color: Tuple[int, int, int], thickness: int) -> np.ndarray:
    """
    Draws a 2D bounding box on an image.
    box_2d: (min_u, min_v, max_u, max_v)
    img: image array.
    color: BGR tuple.
    thickness: line thickness.
    """
    draw_img = img.copy()
    min_u, min_v, max_u, max_v = map(int, box_2d)
    cv2.rectangle(draw_img, (min_u, min_v), (max_u, max_v), color, thickness)
    return draw_img


def draw_rotated_box_2d(img: np.ndarray, x_center: float, y_center: float, width: float, length: float, angle: float, color: Tuple[int, int, int], thickness: int) -> None:
    """
    Draws a rotated 2D bounding box on an image (for BEV visualization usually).
    img: The image/BEV map to draw on.
    x_center, y_center: Center of the box in pixel coordinates.
    width, length: Dimensions of the box.
    angle: Rotation angle in radians.
    color: BGR tuple.
    thickness: Line thickness.
    """
    # Create a rotated rectangle
    # rect expects (center_x, center_y), (width, height), angle_in_degrees
    # Note: OpenCV's angle is typically anti-clockwise from the positive x-axis.
    # You might need to adjust 'angle' based on your coordinate system.
    rect = ((x_center, y_center), (length, width), np.degrees(angle))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, color, thickness)


def gen_hm_radius(heatmap: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """
    Generates a Gaussian heatmap.
    heatmap: 2D numpy array (e.g., single channel of a heatmap).
    center: (x, y) coordinates of the center in pixel space.
    radius: radius of the Gaussian.
    Returns: The updated heatmap with the Gaussian applied.
    """
    center_x, center_y = int(center[0]), int(center[1])
    h, w = heatmap.shape

    # Create a meshgrid for Gaussian calculation
    x, y = np.ogrid[0:h, 0:w]

    # Gaussian formula: exp(-((x - center_x)^2 + (y - center_y)^2) / (2 * radius^2))
    # Note: np.ogrid gives row, col. For (x,y) coordinates, it's (col_idx, row_idx) or (width, height)
    # Adjusting for (center_x, center_y) as (column, row)
    dist_sq = (x - center_y)**2 + (y - center_x)**2
    sigma_sq = 2 * radius**2

    gaussian = np.exp(-dist_sq / sigma_sq)

    # Apply the Gaussian to the heatmap, taking the maximum at each pixel
    # This ensures that overlapping Gaussians contribute correctly.
    heatmap = np.maximum(heatmap, gaussian)
    return heatmap