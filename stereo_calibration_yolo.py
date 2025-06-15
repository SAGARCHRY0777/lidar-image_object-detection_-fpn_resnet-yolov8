import cv2
import numpy as np
import os
from ultralytics import YOLO

# --- Configuration ---
# It's good practice to make paths configurable or use environment variables.
# For this example, we'll keep your original path structure.
KITTI_DATASET_PATH = 'D:\\spa\\SFA3D\\dataset\\kitti\\training'
YOLOV8_MODEL_PATH = 'D:\\spa\\SFA3D\\sfa\\models\\yolov8n.pt'
CALIBRATION_OUTPUT_DIR = os.path.join(KITTI_DATASET_PATH, 'calibrated_parameters')
NUM_IMAGES_TO_CALIBRATE = 10 # Number of initial images to perform calibration
NUM_IMAGES_TO_PREDICT = 20   # Total number of images to process for prediction (including calibrated ones)

# Create output directory for calibrated parameters if it doesn't exist
os.makedirs(CALIBRATION_OUTPUT_DIR, exist_ok=True)

# --- Helper Functions (for KITTI data loading and calibration) ---

def load_kitti_image_paths(dataset_path, num_images):
    """
    Loads a sorted list of image paths for left and right stereo pairs
    from the KITTI dataset. It expects 'image_2' for left and 'image_3' for right.
    
    Args:
        dataset_path (str): Base path to the KITTI training dataset.
        num_images (int): Maximum number of image paths to load.
        
    Returns:
        tuple: (list_left_image_paths, list_right_image_paths)
               Returns empty lists if directories are not found.
    """
    left_image_dir = os.path.join(dataset_path, 'image_2')
    right_image_dir = os.path.join(dataset_path, 'image_3')
    
    # Ensure image directories exist before attempting to list files
    if not os.path.exists(left_image_dir):
        print(f"Error: Left image directory not found: {left_image_dir}")
        return [], []
    if not os.path.exists(right_image_dir):
        print(f"Error: Right image directory not found: {right_image_dir}")
        return [], []

    # Get sorted list of PNG files in each directory
    left_image_files = sorted([os.path.join(left_image_dir, f) for f in os.listdir(left_image_dir) if f.endswith('.png')])
    right_image_files = sorted([os.path.join(right_image_dir, f) for f in os.listdir(right_image_dir) if f.endswith('.png')])
    
    # Return a slice of the lists up to num_images
    return left_image_files[:num_images], right_image_files[:num_images]

def load_kitti_calib_file(dataset_path, image_idx):
    """
    Loads specific calibration parameters from a KITTI 'calib' file for a given image index.
    The calib files are named like '000000.txt' and are expected in 'dataset_path/calib/'.
    It specifically loads P0, P1, P2, P3 (projection matrices), R0_rect (rectification matrix),
    Tr_velo_to_cam (lidar to camera transform), and Tr_imu_to_velo (IMU to lidar transform).
    
    Args:
        dataset_path (str): Base path to the KITTI training dataset.
        image_idx (int): The index of the image (e.g., 0 for '000000.txt').
        
    Returns:
        tuple: (K_left, K_right, R0_rect, Tr_velo_to_cam, P0, P1, P2, P3, Tr_imu_to_velo)
               Returns None for any parameter if the file cannot be loaded or is incomplete/malformed.
    """
    calib_file_path = os.path.join(dataset_path, 'calib', f'{image_idx:06d}.txt')
    
    if not os.path.exists(calib_file_path):
        print(f"Error: Calibration file not found at {calib_file_path}")
        return None, None, None, None, None, None, None, None, None

    calib_params = {}
    with open(calib_file_path, 'r') as f:
        for line in f:
            line = line.strip() # Remove leading/trailing whitespace
            if not line: # Skip empty lines
                continue
            
            if ':' not in line: # Skip lines without a colon (e.g., comments)
                continue

            key, value_str = line.split(':', 1)
            key = key.strip()
            try:
                # Convert the space-separated string of values into a NumPy array of floats
                value = np.array([float(x) for x in value_str.strip().split(' ')])
            except ValueError:
                print(f"Warning: Could not parse calibration value for key '{key}' in {calib_file_path}. Skipping this line.")
                continue # Skip this line if values are malformed
            calib_params[key] = value
    
    # Define required keys and their expected flattened sizes for validation
    required_keys_and_sizes = {
        'P0': 12, 'P1': 12, 'P2': 12, 'P3': 12,
        'R0_rect': 9, 'Tr_velo_to_cam': 12, 'Tr_imu_to_velo': 12
    }
    
    # Validate that all required keys exist and have the correct number of elements
    for key, expected_size in required_keys_and_sizes.items():
        if key not in calib_params or calib_params[key].size != expected_size:
            print(f"Error: Missing or malformed required calibration parameter '{key}' in {calib_file_path}.")
            return None, None, None, None, None, None, None, None, None

    # Reshape the loaded 1D arrays into their respective 2D matrices
    P0 = calib_params['P0'].reshape(3, 4)
    P1 = calib_params['P1'].reshape(3, 4)
    P2 = calib_params['P2'].reshape(3, 4)
    P3 = calib_params['P3'].reshape(3, 4)
    
    R0_rect = calib_params['R0_rect'].reshape(3, 3)
    Tr_velo_to_cam = calib_params['Tr_velo_to_cam'].reshape(3, 4)
    Tr_imu_to_velo = calib_params['Tr_imu_to_velo'].reshape(3, 4)
    
    # Extract intrinsic matrices K_left (from P2) and K_right (from P3).
    # In KITTI, P2 and P3 are camera projection matrices; their top-left 3x3 block is the intrinsic matrix K.
    K_left = P2[:, :3]
    K_right = P3[:, :3]
    
    return K_left, K_right, R0_rect, Tr_velo_to_cam, P0, P1, P2, P3, Tr_imu_to_velo

def perform_targetless_stereo_calibration(img_left, img_right, K_left, K_right):
    """
    Performs targetless stereo calibration to estimate the relative rotation (R) and
    translation (t) between the left and right cameras using feature matching.
    
    Steps:
    1. Feature Detection: Uses ORB (Oriented FAST and Rotated BRIEF) for speed and robustness.
    2. Feature Matching: Employs a Brute-Force Matcher with Hamming distance for ORB descriptors.
    3. Fundamental Matrix Estimation: Uses RANSAC (RANdom SAmple Consensus) to robustly
       estimate the Fundamental Matrix, rejecting outliers.
    4. Essential Matrix Estimation: Derived from the Fundamental Matrix and camera intrinsics.
    5. Camera Pose Estimation: Decomposes the Essential Matrix to recover R and t,
       and performs a cheirality check (points must be in front of both cameras) for validation.
    
    Args:
        img_left (np.array): Left stereo image (grayscale).
        img_right (np.array): Right stereo image (grayscale).
        K_left (np.array): Intrinsic matrix of the left camera (3x3).
        K_right (np.array): Intrinsic matrix of the right camera (3x3).

    Returns:
        tuple: (R, t) - Relative rotation matrix (3x3) and translation vector (3x1).
                       Returns (None, None) if calibration fails or is deemed unreliable.
    """
    
    # 1. Feature Detection and Description using ORB
    # Increased features for potentially better matches, adjusted parameters for quality.
    orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, 
                         WTA_K=2, scoreType=cv2.ORB_FAST_SCORE, patchSize=31, fastThreshold=20)
    
    kp_left, des_left = orb.detectAndCompute(img_left, None)
    kp_right, des_right = orb.detectAndCompute(img_right, None)

    # Check if enough features were detected in both images
    if des_left is None or des_right is None or len(kp_left) < 50 or len(kp_right) < 50:
        print("Warning: Not enough features detected for reliable calibration (need at least 50 in each image). Skipping.")
        return None, None
        
    # 2. Feature Matching using Brute-Force Matcher
    # crossCheck=True ensures that a match (A->B) also has a reverse match (B->A),
    # significantly reducing false positives.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_left, des_right)
    
    # Sort matches by their distance (lower distance indicates better match)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract the matched keypoints' pixel coordinates
    # reshape(-1, 1, 2) is needed for OpenCV functions that expect this format
    pts_left = np.float32([kp_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    if len(pts_left) < 8: # Minimum 8 points are required for Fundamental Matrix estimation
        print(f"Not enough matches ({len(pts_left)}) for Fundamental Matrix estimation (minimum 8 required).")
        return None, None

    # 3. Fundamental Matrix Estimation with RANSAC
    # FM_RANSAC method is used for robust estimation against outliers.
    # The parameters 3.0 and 0.99 are the 'ransacReprojThreshold' and 'confidence' respectively.
    # These are passed as positional arguments, as keyword arguments like 'param1' and 'param2'
    # are not always supported or have changed in different OpenCV versions.
    F, mask_F = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC, 3.0, 0.99)
    
    if F is None or F.shape != (3, 3):
        print("Warning: Fundamental Matrix estimation failed or returned an invalid matrix.")
        return None, None

    # Filter keypoints to keep only the inliers (points consistent with the Fundamental Matrix)
    pts_left_inliers_F = pts_left[mask_F.ravel() == 1]
    pts_right_inliers_F = pts_right[mask_F.ravel() == 1]
    
    if len(pts_left_inliers_F) < 8:
        print(f"Not enough inlier matches ({len(pts_left_inliers_F)}) after RANSAC for Fundamental Matrix.")
        return None, None

    # 4. Essential Matrix Estimation
    # The Essential Matrix (E) relates image points in a calibrated stereo pair.
    # It is derived from the Fundamental Matrix and intrinsic camera parameters.
    # `threshold=1.0` means points within 1.0 pixel of the epipolar line are considered inliers.
    E, mask_E = cv2.findEssentialMat(pts_left_inliers_F, pts_right_inliers_F, K_left, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    if E is None or E.shape != (3, 3):
        print("Warning: Essential Matrix estimation failed or returned an invalid matrix.")
        return None, None
    
    # Filter keypoints to keep only the inliers from Essential Matrix estimation
    pts_left_inliers_E = pts_left_inliers_F[mask_E.ravel() == 1]
    pts_right_inliers_E = pts_right_inliers_F[mask_E.ravel() == 1]

    if len(pts_left_inliers_E) < 8:
        print(f"Not enough inlier matches ({len(pts_left_inliers_E)}) after RANSAC for Essential Matrix.")
        return None, None

    # 5. Camera Pose Estimation (Decomposition of Essential Matrix)
    # `recoverPose` decomposes the Essential Matrix into rotation (R) and translation (t)
    # and attempts to select the physically plausible solution (cheirality check - points in front of cameras).
    _, R, t, _ = cv2.recoverPose(E, pts_left_inliers_E, pts_right_inliers_E, K_left)
    
    # Basic validation: Check if R and t were successfully recovered
    if R is None or t is None:
        print("Warning: Pose recovery failed (R or t is None).")
        return None, None
    
    # Validate R: A valid rotation matrix must be orthogonal and have a determinant of +1.
    if not (np.isclose(np.linalg.det(R), 1.0) and np.allclose(R @ R.T, np.eye(3))):
        print("Warning: Recovered R matrix is not a valid rotation matrix. Pose might be unreliable.")
        return None, None

    # --- Robustness Check: Cheirality (Positive Depth) ---
    # This is a critical step to ensure the estimated R, t define a valid 3D scene.
    # We triangulate the 2D inlier points to 3D and check if they appear in front of both cameras.
    
    # Construct projection matrices for triangulation
    # P_left = K_left @ [I | 0] (Camera 1 is at origin, looking along Z)
    P_left = K_left @ np.hstack((np.eye(3), np.zeros((3, 1))))
    # P_right = K_right @ [R | t] (Camera 2's pose relative to Camera 1)
    P_right = K_right @ np.hstack((R, t))

    # Triangulate 2D points to 3D points in homogeneous coordinates
    # pts_left_inliers_E and pts_right_inliers_E are already in the expected format (N, 1, 2)
    points4D_hom = cv2.triangulatePoints(P_left, P_right, pts_left_inliers_E, pts_right_inliers_E)
    
    # Convert homogeneous 4D points to non-homogeneous 3D points
    # Divide X, Y, Z by the fourth component (W)
    points3D = points4D_hom / points4D_hom[3, :] # points3D will be (4, N)

    # Check depth for left camera (Z-coordinate of 3D points in left camera's frame)
    depth_left_cam = points3D[2, :] # Z-coordinate is the third row (index 2)

    # Check depth for right camera. Need to transform 3D points (from left camera frame)
    # to the right camera's coordinate system using R and t.
    points3D_in_right_cam_frame = R @ points3D[:3, :] + t # points3D[:3, :] gets X, Y, Z from 3D points
    depth_right_cam = points3D_in_right_cam_frame[2, :]

    # Count points that have positive depth (in front of the camera) in both views
    num_positive_depth_left = np.sum(depth_left_cam > 0)
    num_positive_depth_right = np.sum(depth_right_cam > 0)
    
    # Define a minimum ratio of points that must have positive depth
    min_positive_depth_ratio = 0.70 # At least 70% of the triangulated points should have positive depth

    # If the ratio of points with positive depth is too low, the pose estimate is likely incorrect.
    if (num_positive_depth_left / len(pts_left_inliers_E)) < min_positive_depth_ratio or \
       (num_positive_depth_right / len(pts_right_inliers_E)) < min_positive_depth_ratio:
        print(f"Warning: Insufficient positive depth points after pose recovery (Left: {num_positive_depth_left}/{len(pts_left_inliers_E)}, "
              f"Right: {num_positive_depth_right}/{len(pts_right_inliers_E)}). Pose might be unreliable. Setting R, t to None.")
        return None, None # Indicate calibration failure
    
    print(f"Successfully estimated pose with {len(pts_left_inliers_E)} inliers and passed cheirality check.")
    return R, t

def format_matrix_for_kitti(matrix, prefix):
    """
    Formats a NumPy matrix into a single line string with a given prefix,
    using scientific notation for floats, similar to the KITTI calibration file format.
    """
    flattened = matrix.flatten()
    # Format each float to 12 decimal places using scientific notation ('e')
    formatted_values = [f"{x:.12e}" for x in flattened]
    return f"{prefix}: {' '.join(formatted_values)}"

def save_calibration_to_kitti_txt(filepath, P0_orig, P1_orig, P2_orig, P3_orig, R0_rect_orig, Tr_velo_to_cam_orig, Tr_imu_to_velo_orig, R_stereo_estimated=None, t_stereo_estimated=None):
    """
    Saves calibration parameters to a TXT file in KITTI format.
    It saves all original KITTI parameters (P0-P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo).
    Optionally, it includes the newly estimated R_stereo and t_stereo as comments,
    distinguishing them from the original KITTI data.
    
    Args:
        filepath (str): The full path to the output calibration file.
        P0_orig, P1_orig, P2_orig, P3_orig (np.array): Original KITTI projection matrices.
        R0_rect_orig (np.array): Original KITTI rectification matrix.
        Tr_velo_to_cam_orig (np.array): Original KITTI Velodyne-to-camera transform.
        Tr_imu_to_velo_orig (np.array): Original KITTI IMU-to-Velodyne transform.
        R_stereo_estimated (np.array, optional): Estimated stereo rotation matrix.
        t_stereo_estimated (np.array, optional): Estimated stereo translation vector.
    """
    with open(filepath, 'w') as f:
        # Write the original KITTI calibration parameters.
        # These are usually standard for the dataset and represent the ground truth setup.
        f.write(format_matrix_for_kitti(P0_orig, 'P0') + "\n")
        f.write(format_matrix_for_kitti(P1_orig, 'P1') + "\n")
        f.write(format_matrix_for_kitti(P2_orig, 'P2') + "\n")
        f.write(format_matrix_for_kitti(P3_orig, 'P3') + "\n")
        f.write(format_matrix_for_kitti(R0_rect_orig, 'R0_rect') + "\n")
        f.write(format_matrix_for_kitti(Tr_velo_to_cam_orig, 'Tr_velo_to_cam') + "\n")
        f.write(format_matrix_for_kitti(Tr_imu_to_velo_orig, 'Tr_imu_to_velo') + "\n")

        # Add the newly estimated stereo parameters as comments.
        # These represent the relative pose between the left and right cameras,
        # which was calculated from feature matching.
        if R_stereo_estimated is not None and t_stereo_estimated is not None:
            f.write("\n# --- Estimated Stereo Calibration (Left to Right Camera) ---\n")
            f.write(f"# R_stereo (estimated): {' '.join([f'{x:.12e}' for x in R_stereo_estimated.flatten()])}\n")
            f.write(f"# t_stereo (estimated): {' '.join([f'{x:.12e}' for x in t_stereo_estimated.flatten()])}\n")
            f.write("# --- End of Estimated Stereo Calibration ---\n")
        
    print(f"Saved calibration file: {os.path.basename(filepath)}")

# --- Main Calibration Loop ---
print("Initiating targetless stereo calibration process...")

# Load image paths for the calibration phase
left_image_paths_calib, right_image_paths_calib = load_kitti_image_paths(KITTI_DATASET_PATH, NUM_IMAGES_TO_CALIBRATE)

# Check if image paths were successfully loaded
if not left_image_paths_calib or not right_image_paths_calib or \
   len(left_image_paths_calib) != len(right_image_paths_calib):
    print("Error: Failed to load enough stereo image pairs for calibration. Please check dataset path and image directories.")
    exit()

# Dictionary to store successfully calibrated stereo parameters for later use in prediction
# Key: image index, Value: dictionary containing 'R', 't', 'K_left', 'K_right'
calibrated_params = {} 

for i in range(NUM_IMAGES_TO_CALIBRATE):
    print(f"\nProcessing calibration for image pair {i+1}/{NUM_IMAGES_TO_CALIBRATE} (Index: {i:06d})...")
    
    # Load all original KITTI calibration parameters for the current image index.
    # These are crucial for intrinsic matrices and for saving the complete calibration file.
    K_left_orig, K_right_orig, R0_rect_orig, Tr_velo_to_cam_orig, P0_orig, P1_orig, P2_orig, P3_orig, Tr_imu_to_velo_orig = load_kitti_calib_file(KITTI_DATASET_PATH, i)
    
    if K_left_orig is None:
        print(f"Skipping calibration for pair {i} due to issues with original KITTI calibration file. "
              f"Ensure '{os.path.join(KITTI_DATASET_PATH, 'calib', f'{i:06d}.txt')}' exists and is well-formed.")
        # Attempt to save original KITTI file even if intrinsics extraction fails for completeness
        # but don't try stereo calibration without valid intrinsics.
        calib_filename = f"{i:06d}.txt"
        calib_filepath = os.path.join(CALIBRATION_OUTPUT_DIR, calib_filename)
        # Check if original parameters are None before attempting to save.
        # If any are None due to load_kitti_calib_file failing, then cannot save.
        if all(x is not None for x in [P0_orig, P1_orig, P2_orig, P3_orig, R0_rect_orig, Tr_velo_to_cam_orig, Tr_imu_to_velo_orig]):
             save_calibration_to_kitti_txt(calib_filepath, P0_orig, P1_orig, P2_orig, P3_orig, R0_rect_orig, Tr_velo_to_cam_orig, Tr_imu_to_velo_orig)
        continue

    # Load grayscale images for feature detection and matching
    img_left = cv2.imread(left_image_paths_calib[i], cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_image_paths_calib[i], cv2.IMREAD_GRAYSCALE)
    
    if img_left is None or img_right is None:
        print(f"Error: Could not load grayscale images for pair {i} from paths: "
              f"'{left_image_paths_calib[i]}' and '{right_image_paths_calib[i]}'. Skipping.")
        # Still save the original KITTI calib for completeness if image loading fails
        calib_filename = f"{i:06d}.txt"
        calib_filepath = os.path.join(CALIBRATION_OUTPUT_DIR, calib_filename)
        save_calibration_to_kitti_txt(calib_filepath, P0_orig, P1_orig, P2_orig, P3_orig, R0_rect_orig, Tr_velo_to_cam_orig, Tr_imu_to_velo_orig)
        continue

    # Perform targetless stereo calibration using the loaded original intrinsic matrices.
    # This function returns the estimated relative pose (R_stereo, t_stereo).
    R_stereo, t_stereo = perform_targetless_stereo_calibration(img_left, img_right, K_left_orig, K_right_orig)
    
    calib_filename = f"{i:06d}.txt" # Standard KITTI naming convention for output files
    calib_filepath = os.path.join(CALIBRATION_OUTPUT_DIR, calib_filename)
    
    if R_stereo is not None and t_stereo is not None:
        # If stereo pose estimation was successful, save the original KITTI parameters
        # and include our estimated R_stereo and t_stereo as comments in the output file.
        save_calibration_to_kitti_txt(calib_filepath, P0_orig, P1_orig, P2_orig, P3_orig, 
                                      R0_rect_orig, Tr_velo_to_cam_orig, Tr_imu_to_velo_orig, 
                                      R_stereo_estimated=R_stereo, t_stereo_estimated=t_stereo)
        
        # Store the estimated R and t, along with the original intrinsics, for later use in prediction.
        calibrated_params[i] = {
            'R': R_stereo,
            't': t_stereo,
            'K_left': K_left_orig, # Keep original intrinsics for consistent projection
            'K_right': K_right_orig
        }
    else:
        print(f"Failed to estimate stereo pose for image pair {i}. Saving original KITTI calib only for completeness.")
        # If estimation fails, still save the original KITTI calib file, but without the estimated stereo parameters.
        save_calibration_to_kitti_txt(calib_filepath, P0_orig, P1_orig, P2_orig, P3_orig, 
                                      R0_rect_orig, Tr_velo_to_cam_orig, Tr_imu_to_velo_orig)

print("\nStereo calibration phase complete.")
if not calibrated_params:
    print("Warning: No successful stereo calibrations were performed. YOLOv8 predictions will run without custom stereo context.")

# --- YOLOv8 Prediction with Dynamic Calibration (Interactive Display) ---

print("\nStarting YOLOv8 predictions and interactive display...")

try:
    # Load the pre-trained YOLOv8 model
    model = YOLO(YOLOV8_MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLOv8 model from {YOLOV8_MODEL_PATH}: {e}")
    print("Please ensure the model path is correct and the model file exists (e.g., 'yolov8n.pt').")
    exit()

# Create a resizable window to display predictions
cv2.namedWindow("YOLOv8 Predictions", cv2.WINDOW_NORMAL) 

# Load all left image paths for the prediction phase
left_image_paths_all, _ = load_kitti_image_paths(KITTI_DATASET_PATH, NUM_IMAGES_TO_PREDICT)

if not left_image_paths_all:
    print("Error: No images found for prediction. Please verify KITTI_DATASET_PATH and 'image_2' directory.")
    cv2.destroyAllWindows()
    exit()

current_image_idx = 0

# Loop through images for prediction
while current_image_idx < NUM_IMAGES_TO_PREDICT:
    # Ensure we don't go out of bounds of the loaded image paths
    if current_image_idx >= len(left_image_paths_all):
        print("Reached end of image list for prediction.")
        break

    print(f"\nProcessing image {current_image_idx+1}/{NUM_IMAGES_TO_PREDICT} (Index: {current_image_idx:06d})...")

    img_left_path = left_image_paths_all[current_image_idx]
    img_left_color = cv2.imread(img_left_path) # Load as color image for YOLOv8 display
    
    if img_left_color is None:
        print(f"Error: Could not load left image for prediction at index {current_image_idx} from '{img_left_path}'. Skipping.")
        current_image_idx += 1
        continue

    # Initialize calibration parameters for the current image.
    # These will be populated based on whether we have estimated parameters or fall back to original KITTI.
    K_left_for_pred, K_right_for_pred, R_stereo_for_pred, t_stereo_for_pred = None, None, None, None

    # Attempt to retrieve our estimated stereo calibration parameters for this specific image index.
    if current_image_idx in calibrated_params:
        current_calib = calibrated_params[current_image_idx]
        R_stereo_for_pred = current_calib['R']
        t_stereo_for_pred = current_calib['t']
        K_left_for_pred = current_calib['K_left']
        K_right_for_pred = current_calib['K_right']
        print(f"    Using **estimated stereo pose** and **original KITTI intrinsics** for image {current_image_idx}.")
    else:
        # If our estimation was not performed for this image index (e.g., beyond NUM_IMAGES_TO_CALIBRATE)
        # or if it failed for this specific image, try loading only the original KITTI calibration file.
        # This will provide at least the intrinsic matrices.
        K_left_orig, K_right_orig, _, _, _, _, _, _, _ = load_kitti_calib_file(KITTI_DATASET_PATH, current_image_idx)
        if K_left_orig is not None:
            K_left_for_pred = K_left_orig
            K_right_for_pred = K_right_orig
            # If no estimated stereo pose, we use identity for R and zero for t.
            # This effectively means no relative transformation is applied when just using intrinsics.
            R_stereo_for_pred = np.eye(3) 
            t_stereo_for_pred = np.zeros((3,1)) 
            print(f"    Using **original KITTI intrinsics** (no estimated stereo pose from this script) for image {current_image_idx}.")
        else:
            print(f"    No usable calibration data (neither estimated nor original KITTI) available for image {current_image_idx}. Proceeding with YOLOv8 only (no stereo context).")
            # If no calibration data at all, the variables will remain None, and stereo context won't be displayed.
            
    # Perform YOLOv8 prediction on the loaded left image
    results = model(img_left_color, verbose=False) # verbose=False suppresses console output for each prediction

    # Iterate through detected objects and draw bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates (x1, y1, x2, y2), confidence, and class ID
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            # Draw the bounding box
            cv2.rectangle(img_left_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Create a label with class name and confidence
            label = f"{model.names[cls]} {conf:.2f}"
            # Put the label text above the bounding box
            cv2.putText(img_left_color, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # --- Conceptual Integration with Stereo Calibration ---
            # This section explains where the calibrated stereo parameters would be utilized
            # for full 3D object perception, which goes beyond 2D YOLOv8 detection.
            #
            # For comprehensive 3D understanding (e.g., precise object depth and 3D location),
            # you would typically perform the following steps:
            # 1. Image Rectification: Use K_left_for_pred, K_right_for_pred, R_stereo_for_pred,
            #    and t_stereo_for_pred to rectify both left and right stereo images.
            #    `cv2.stereoRectify` is used for this to align the images so epipolar lines are horizontal.
            # 2. Stereo Matching: Apply a stereo matching algorithm (e.g., `cv2.StereoBM` or `cv2.StereoSGBM`)
            #    on the rectified image pair to compute a disparity map. The disparity (d) for a pixel
            #    is the horizontal shift between its location in the left and right images.
            # 3. 3D Triangulation: For each detected object (or pixel within it), if you have its disparity,
            #    you can triangulate its 3D position (X, Y, Z) in space. The depth (Z) can be
            #    approximated by the formula: Z = (focal_length * baseline) / disparity,
            #    where 'focal_length' is from K_left_for_pred and 'baseline' is derived from t_stereo_for_pred.
            #
            # In this script, we are primarily demonstrating 2D object detection with YOLOv8 and
            # the estimation/loading of stereo calibration parameters. The actual 3D reconstruction
            # is not performed but is conceptually explained.
            
            # Print the relevant calibration info if available for context
            if R_stereo_for_pred is not None and t_stereo_for_pred is not None:
                print(f"        Detected {model.names[cls]}: Bounding box ({x1},{y1})-({x2},{y2})")
                print(f"        Stereo R (estimated/fallback): \n{R_stereo_for_pred.round(4)}\n        Stereo t (estimated/fallback): \n{t_stereo_for_pred.T.round(4)}")
            if K_left_for_pred is not None:
                print(f"        Left Camera Intrinsics K_left (original KITTI/fallback):\n{K_left_for_pred.round(2)}")

    # Display the image with YOLOv8 predictions
    cv2.imshow("YOLOv8 Predictions", img_left_color)
    
    # Wait for a key press for interactive navigation
    key = cv2.waitKey(0) # Waits indefinitely until a key is pressed
    
    if key == ord('n'): # Press 'n' to go to the next image
        current_image_idx += 1
    elif key == ord('p'): # Press 'p' to go to the previous image
        current_image_idx = max(0, current_image_idx - 1) # Ensure index doesn't go below 0
    elif key == ord('q'): # Press 'q' to quit the application
        break

cv2.destroyAllWindows()
print("\nYOLOv8 prediction process complete. Application closed.")
