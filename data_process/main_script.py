# main_script.py (example of how to use)
import numpy as np
from bev_projection_utils import example_usage_bev_projection
from pathlib import Path

# --- Dummy Data for Demonstration ---
# In a real scenario, you would load your actual LiDAR and bounding box data here.

# Dummy LiDAR points (Nx4: x, y, z, intensity)
# Create some random points resembling a ground plane and some objects
np.random.seed(42) # for reproducibility
num_lidar_points = 5000
dummy_lidar_points = np.random.rand(num_lidar_points, 4)
# Scale and offset to simulate a road segment
dummy_lidar_points[:, 0] = (dummy_lidar_points[:, 0] * 100) - 50 # x from -50 to 50
dummy_lidar_points[:, 1] = (dummy_lidar_points[:, 1] * 20) - 10  # y from -10 to 10
dummy_lidar_points[:, 2] = (dummy_lidar_points[:, 2] * 0.5) - 1.5 # z around ground level (-1.5 to -1.0)
dummy_lidar_points[:, 3] = np.random.rand(num_lidar_points) # intensity

# Add some points for dummy objects (e.g., higher z-values)
object_points = np.random.rand(500, 4)
object_points[:, 0] = np.random.uniform(5, 7, 500) # x between 5 and 7
object_points[:, 1] = np.random.uniform(1, 3, 500)  # y between 1 and 3
object_points[:, 2] = np.random.uniform(-0.5, 1.0, 500) # z for an object
object_points[:, 3] = np.random.rand(500)
dummy_lidar_points = np.vstack((dummy_lidar_points, object_points))


# Dummy Bounding Box data (replace with your actual detections/annotations)
dummy_box_data = [
    {'center': [5.0, 2.0, -1.0], 'dims': [4.5, 1.8, 1.7], 'heading': np.radians(30)}, # A car
    {'center': [-10.0, -5.0, -1.0], 'dims': [3.0, 1.5, 1.6], 'heading': np.radians(-60)}, # Another car
    {'center': [0.0, 8.0, -1.0], 'dims': [0.8, 0.8, 1.8], 'heading': np.radians(90)} # A pedestrian
]

# Path to your calibration file
# Make sure you have a 'calibration.json' file in the same directory
# or provide the correct path to your calibration file.
calibration_file_path = Path("./calibration.json")

# Create a dummy calibration.json for testing if you don't have one
if not calibration_file_path.exists():
    dummy_calib_content = {
        "camera_data": [
            {
                "key": "front_center",
                "value": {
                    "focal_length_x_px_": 1000.0,
                    "focal_length_y_px_": 1000.0,
                    "focal_center_x_px_": 960.0,
                    "focal_center_y_px_": 600.0,
                    "skew_": 0.0,
                    "distortion_coeffs": [0.0, 0.0, 0.0],
                    "vehicle_SE3_camera_": {
                        "translation": [1.5, 0.0, 1.5],
                        "rotation": {"coefficients": [0.0, 0.0, 0.0, 1.0]} # qw=1 for identity rotation
                    }
                }
            }
        ],
        "lidar_data": [
            {
                "key": "down_lidar",
                "value": {
                    "vehicle_SE3_down_lidar_": {
                        "translation": [0.0, 0.0, 0.0], # LiDAR at vehicle origin for simplicity
                        "rotation": {"coefficients": [0.0, 0.0, 0.0, 1.0]} # Identity rotation
                    }
                }
            }
        ]
    }
    import json
    with open(calibration_file_path, "w") as f:
        json.dump(dummy_calib_content, f, indent=4)
    print(f"Created a dummy calibration file at: {calibration_file_path}")


# Run the BEV projection example
example_usage_bev_projection(
    str(calibration_file_path), # Convert Path object to string
    dummy_lidar_points,
    dummy_box_data
)