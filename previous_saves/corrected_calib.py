# corrected_calib.py (your main file, using the new utilities)
import json
import numpy as np
from typing import Any, Dict, List, NamedTuple, Tuple, Union
from pathlib import Path    
# Import the custom SE3 and quat2rotmat implementations
from new_se3 import SE3
from new_transform_utils import quat2rotmat

# Placeholder for camera list and image dimensions (since argoverse.utils.camera_stats is not used)
# You MUST populate these based on your specific camera setup.
CAMERA_LIST = [
    "front_left", "front_right", "front_center",
    "side_left", "side_right",
    "rear_left", "rear_right"
]
RECTIFIED_STEREO_CAMERA_LIST = ["rectified_front_left", "rectified_front_right"] # Example

# This function needs to be adapted based on your actual image dimensions.
# In the original argoverse, this comes from a utility.
def get_image_dims_for_camera(camera_name: str) -> Tuple[int, int]:
    """
    Placeholder for image dimensions. You need to provide actual dimensions.
    """
    # Example dimensions, replace with your actual camera resolutions
    if "front" in camera_name:
        return 1920, 1200 # Example: Full HD width, 1200 height
    elif "side" in camera_name or "rear" in camera_name:
        return 1280, 960 # Example: Common resolution for side/rear cameras
    else:
        raise ValueError(f"Unknown camera name for dimensions: {camera_name}")

class CameraConfig(NamedTuple):
    """Camera config for extrinsic matrix, intrinsic matrix, image width/height.
    Args:
        extrinsic: extrinsic matrix
        intrinsic: intrinsic matrix
        img_width: image width
        img_height: image height
    """
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    img_width: int
    img_height: int
    distortion_coeffs: np.ndarray


def get_camera_extrinsic_matrix(config_data: Dict[str, Any]) -> np.ndarray:
    """
    Load camera calibration rotation and translation to form the extrinsic matrix (camera_SE3_egovehicle).
    This matrix transforms points from the ego-vehicle frame to the camera frame.
    The input `config_data` is the 'value' dictionary for a specific camera from the calibration JSON.
    """
    egovehicle_SE3_camera = config_data["vehicle_SE3_camera_"]
    egovehicle_t_camera = np.array(egovehicle_SE3_camera["translation"], dtype=np.float32)
    egovehicle_q_camera = egovehicle_SE3_camera["rotation"]["coefficients"] # (qx, qy, qz, qw)

    # Use the custom quat2rotmat, which expects (w, x, y, z)
    # Reorder from (qx, qy, qz, qw) to (qw, qx, qy, qz)
    R_camera_egovehicle = quat2rotmat(np.array([egovehicle_q_camera[3], egovehicle_q_camera[0], egovehicle_q_camera[1], egovehicle_q_camera[2]]))

    # Build the SE3 transform
    T_camera_egovehicle = SE3(rotation=R_camera_egovehicle, translation=egovehicle_t_camera)
    return T_camera_egovehicle.transform_matrix


def get_camera_intrinsic_matrix(camera_config_data: Dict[str, Any]) -> np.ndarray:
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


def get_calibration_config(calib_data: Dict[str, Any], camera_name: str) -> CameraConfig:
    """Helper to get a CameraConfig object for a given camera name."""
    camera_data = None
    for cam in calib_data['camera_data']:
        if cam['key'] == camera_name:
            camera_data = cam['value']
            break
    if camera_data is None:
        raise ValueError(f"Calibration data for camera {camera_name} not found.")

    extrinsic_matrix = get_camera_extrinsic_matrix(camera_data)
    intrinsic_matrix = get_camera_intrinsic_matrix(camera_data)
    img_width, img_height = get_image_dims_for_camera(camera_name)
    distortion_coeffs = np.array(camera_data.get("distortion_coeffs", [0.0, 0.0, 0.0]), dtype=np.float32)

    return CameraConfig(extrinsic_matrix, intrinsic_matrix, img_width, img_height, distortion_coeffs)


class Calibration:
    """Calibration matrices and utils.

    3d XYZ are in 3D egovehicle coord.
    2d box xy are in image coord, normalized by width and height
    Point cloud are in egovehicle coord

    ::

       xy_image = K * [R|T] * xyz_ego

       xyz_image = [R|T] * xyz_ego

       image coord:
        ----> x-axis (u)
       |
       |
       v y-axis (v)

    egovehicle coord:
    front x, left y, up z
    """

    def __init__(self, camera_config: CameraConfig, calib: Dict[str, Any]) -> None:
        """Create a Calibration instance.

        Args:
            camera_config: A camera config
            calib: Calibration data
        """
        self.camera_config = camera_config

        self.calib_data = calib

        self.extrinsic = self.camera_config.extrinsic
        self.R = self.extrinsic[0:3, 0:3]
        self.T = self.extrinsic[0:3, 3]

        self.K = self.camera_config.intrinsic

        self.cu = self.calib_data["value"]["focal_center_x_px_"]
        self.cv = self.calib_data["value"]["focal_center_y_px_"]
        self.fu = self.calib_data["value"]["focal_length_x_px_"]
        self.fv = self.calib_data["value"]["focal_length_y_px_"]

        # This assumes a standard pinhole model with no principal point offset in K.
        # If K has non-zero K[0,3] or K[1,3] from the original argoverse data, it implies
        # something like a stereo baseline which is not typically part of standard intrinsic.
        # For a standard pinhole, K is 3x3. The original ref_calib.py creates a 3x4 K matrix.
        # The bx and by here are based on the assumption that K is derived from a 3x4 projection matrix.
        # If your K is truly 3x3, then these should be 0.
        self.bx = self.K[0, 3] / (-self.fu) if self.fu != 0 else 0
        self.by = self.K[1, 3] / (-self.fv) if self.fv != 0 else 0


    def project_to_image(self, pts_3d_ego: np.ndarray) -> np.ndarray:
        """Project 3D ego-vehicle coordinates to 2D image coordinates (u, v)."""
        # Convert ego-vehicle points to homogeneous coordinates
        pts_3d_ego_hom = np.hstack([pts_3d_ego, np.ones((pts_3d_ego.shape[0], 1))]).T # (4, N)

        # Transform points from ego-vehicle frame to Camera frame using extrinsic matrix
        # self.extrinsic is camera_SE3_egovehicle (4x4)
        pts_3d_cam_hom = self.extrinsic @ pts_3d_ego_hom # (4, N)

        # Convert to 3D camera coordinates (3, N)
        pts_3d_cam = pts_3d_cam_hom[:3, :]

        # Apply intrinsic matrix to project to 2D image coordinates (3, N)
        uv_hom = self.K @ pts_3d_cam # (3, N)

        # Normalize by depth (Z-coordinate in camera frame)
        depth = uv_hom[2, :]
        
        # Handle points behind the camera or very far away
        valid_indices = depth > 1e-6 # Small positive threshold
        
        uv_norm = np.full(uv_hom.shape, np.nan)
        if np.any(valid_indices):
            uv_norm[:, valid_indices] = uv_hom[:, valid_indices] / depth[valid_indices]

        # Extract u, v coordinates (2, N)
        uv = uv_norm[:2, :].T # (N, 2)
        return uv


def undistort_radius(radius_undist: float, distort_coeffs: List[float]) -> float:
    """
    Performs camera distortion for a single undistorted radius.
    Note that we have 3 distortion parameters.

    Args:
        radius_undist: undistorted radius
        distort_coeffs: list of distortion coefficients

    Returns:
        distortion radius
    """
    radius_dist = radius_undist
    r_u_pow = radius_undist
    for distortion_coefficient in distort_coeffs:
        r_u_pow *= radius_undist ** 2
        radius_dist += r_u_pow * distortion_coefficient

    return radius_dist


def proj_cam_to_uv(
    uv_cam: np.ndarray, camera_config: CameraConfig, remove_nan: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projects points from camera coordinates to 2D image coordinates.
    Args:
        uv_cam: Nx3 array of points in camera coordinates.
        camera_config: CameraConfig object with intrinsic matrix and distortion coefficients.
        remove_nan: If True, remove points that project outside the image or behind the camera.
    Returns:
        uv: Numpy array of shape (N,2) with dtype np.float32
        uv_cam: Numpy array of shape (3,N) with dtype np.float32 (homogeneous camera coords)
        valid_pts_bool: Numpy array of shape (N,) with dtype bool (mask for valid points)
    """
    if uv_cam.shape[1] != 3:
        raise ValueError("uv_cam must be Nx3 array.")

    # Convert to homogeneous camera coordinates
    uv_cam_hom = np.vstack([uv_cam.T, np.ones(uv_cam.shape[0])]) # (4, N) or should be (3,N) if only 3D points
    uv_cam_hom = uv_cam.T # (3, N) assuming uv_cam is already 3D camera coords

    # Project to 2D image coordinates using intrinsic matrix
    uv_proj_hom = camera_config.intrinsic @ uv_cam_hom # (3, N)

    # Normalize by depth (Z-coordinate)
    depth = uv_proj_hom[2, :]
    
    # Handle points behind the camera
    valid_pts_bool = depth > 1e-6 # Threshold for positive depth

    uv_proj_normalized = np.full(uv_proj_hom.shape, np.nan)
    if np.any(valid_pts_bool):
        uv_proj_normalized[:, valid_pts_bool] = uv_proj_hom[:, valid_pts_bool] / depth[valid_pts_bool]

    # Extract u, v coordinates
    uv = uv_proj_normalized[:2, :].T # (N, 2)

    # Apply distortion if coefficients are provided
    if camera_config.distortion_coeffs is not None and len(camera_config.distortion_coeffs) > 0:
        # Calculate radial distance from principal point
        # Assuming principal point is (cu, cv) from intrinsic matrix (K[0,2], K[1,2])
        u_dist_from_center = uv[:, 0] - camera_config.intrinsic[0, 2]
        v_dist_from_center = uv[:, 1] - camera_config.intrinsic[1, 2]
        r_undist = np.sqrt(u_dist_from_center**2 + v_dist_from_center**2)

        # Apply distortion model (radial distortion only for simplicity, assuming no tangential)
        r_distorted = np.array([undistort_radius(r_u, camera_config.distortion_coeffs) for r_u in r_undist])

        # Scale back to distorted coordinates
        scale_factor = np.full_like(r_distorted, np.nan)
        non_zero_r_undist_mask = r_undist != 0
        scale_factor[non_zero_r_undist_mask] = r_distorted[non_zero_r_undist_mask] / r_undist[non_zero_r_undist_mask]
        
        uv_distorted = np.copy(uv)
        uv_distorted[valid_pts_bool, 0] = camera_config.intrinsic[0, 2] + u_dist_from_center[valid_pts_bool] * scale_factor[valid_pts_bool]
        uv_distorted[valid_pts_bool, 1] = camera_config.intrinsic[1, 2] + v_dist_from_center[valid_pts_bool] * scale_factor[valid_pts_bool]
        uv = uv_distorted

    # Filter points outside image boundaries if remove_nan is True
    if remove_nan:
        img_width, img_height = camera_config.img_width, camera_config.img_height
        within_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < img_width) & \
                        (uv[:, 1] >= 0) & (uv[:, 1] < img_height)
        
        # Combine valid_pts_bool with within_bounds
        final_valid_mask = valid_pts_bool & within_bounds
        
        # Apply nan to invalid points for consistency
        uv[~final_valid_mask] = np.nan
        uv_cam_hom[:, ~final_valid_mask] = np.nan # Mark corresponding homogeneous points as nan

        return uv, uv_cam_hom, final_valid_mask
    
    return uv, uv_cam_hom, valid_pts_bool


def project_lidar_to_undistorted_img(
    lidar_points_h: np.ndarray,
    calib_data: Dict[str, Any],
    camera_name: str,
    remove_nan: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, CameraConfig]:
    """
    Projects LiDAR points (homogeneous Nx4) to undistorted image coordinates.
    Returns: uv (Nx2), uv_cam_hom (3xN), valid_pts_bool (N), camera_config
    """
    camera_config = get_calibration_config(calib_data, camera_name)

    # LiDAR to Ego-vehicle transform (vehicle_SE3_down_lidar_)
    lidar_data_raw = calib_data['lidar_data'][0]['value'] # Assuming 'down_lidar' is the first entry
    rot_coeffs_lv = lidar_data_raw['vehicle_SE3_down_lidar_']['rotation']['coefficients'] # (qx, qy, qz, qw)
    trans_coeffs_lv = np.array(lidar_data_raw['vehicle_SE3_down_lidar_']['translation'], dtype=np.float32)
    
    # Reorder for quat2rotmat (qw, qx, qy, qz)
    R_l_v = quat2rotmat(np.array([rot_coeffs_lv[3], rot_coeffs_lv[0], rot_coeffs_lv[1], rot_coeffs_lv[2]]))
    T_l_v = SE3(rotation=R_l_v, translation=trans_coeffs_lv).transform_matrix # Transform from LiDAR to Ego-vehicle

    # Transform LiDAR points to Ego-vehicle frame
    points_ego_hom = T_l_v @ lidar_points_h.T # (4, N)

    # Transform ego-vehicle points to Camera frame using camera_SE3_egovehicle (extrinsic)
    # camera_config.extrinsic is the camera_SE3_egovehicle (4x4)
    uv_cam = camera_config.extrinsic @ points_ego_hom # (4, N)

    # Project from 3D camera coordinates to 2D image coordinates (with distortion)
    # proj_cam_to_uv expects Nx3 camera coordinates, not homogeneous 4xN
    uv_cam_3d = uv_cam[:3, :].T # (N, 3)
    uv, uv_cam_hom, valid_pts_bool = proj_cam_to_uv(uv_cam_3d, camera_config, remove_nan)

    return uv, uv_cam_hom, valid_pts_bool, camera_config


def load_calib(calib_filepath: Union[str, Path]) -> Dict[Any, Calibration]:
    """Load Calibration object for each camera from the calibration filepath Args: calib_filepath (str): path to the calibration file Returns: list of Calibration object for each camera """
    with open(calib_filepath, "r") as f:
        calib = json.load(f)

    calib_list = {}
    for camera in CAMERA_LIST:
        cam_config = get_calibration_config(calib, camera)
        calib_cam = next(
            (c for c in calib["camera_data"] if c["key"] == camera), None
        )
        if calib_cam is None:
            continue
        calib_ = Calibration(cam_config, calib_cam)
        calib_list[camera] = calib_
    return calib_list


def load_stereo_calib(calib_filepath: Union[str, Path]) -> Dict[Any, Calibration]:
    """Load Calibration object for the rectified stereo cameras from the calibration filepath Args: calib_filepath (str): path to the stereo calibration file Returns: list of stereo Calibration object for the rectified stereo cameras """
    with open(calib_filepath, "r") as f:
        calib = json.load(f)

    calib_list = {}
    for camera in RECTIFIED_STEREO_CAMERA_LIST:
        cam_config = get_calibration_config(calib, camera)
        calib_cam = next(
            (c for c in calib["camera_data"] if c["key"] == camera), None
        )
        if calib_cam is None:
            continue
        calib_ = Calibration(cam_config, calib_cam)
        calib_list[camera] = calib_
    return calib_list