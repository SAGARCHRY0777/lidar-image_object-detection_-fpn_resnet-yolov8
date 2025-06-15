# argoverse_config.py
import numpy as np

# Define Argoverse-specific configurations
# This is a simplified example. You might need more detailed bounds
# based on your specific Argoverse dataset and sensor setup.
# These values will significantly impact your BEV map.
BEV_WIDTH = 608  # Example, adjust as needed
BEV_HEIGHT = 608 # Example, adjust as needed
DISCRETIZATION = 0.1 # m/pixel, adjust based on desired resolution

# BEV boundary for Argoverse (example values, tune for your dataset)
# These represent min/max X, Y, Z in the vehicle frame (lidar frame).
# X is usually forward, Y is left, Z is up.
# Argoverse typically has a range of +/- 50m for X/Y.
boundary = {
    "minX": -50,
    "maxX": 50,
    "minY": -50,
    "maxY": 50,
    "minZ": -3,  # Ground level or slightly below
    "maxZ": 5,   # Above ground level, for objects
}

bound_size_x = boundary['maxX'] - boundary['minX']
bound_size_y = boundary['maxY'] - boundary['minY']
bound_size_z = boundary['maxZ'] - boundary['minZ']

# Argoverse classes and their mapping to integer IDs
# This needs to align with your Argoverse annotation parsing.
# Refer to Argoverse documentation for official class names.
CLASS_NAME_TO_ID = {
    'VEHICLE': 0,
    'PEDESTRIAN': 1,
    'BICYCLE': 2,
    #'OTHER_MOVER': -99, # Example: ignore certain classes
    #'ANIMAL': -99,
    #'ON_ROAD_OBSTACLE': -99,
    # Add more classes if needed, or define -99 for ignored classes
}

ID_TO_CLASS_NAME = {v: k for k, v in CLASS_NAME_TO_ID.items()}

# Colors for visualization (RGB)
colors = {
    0: (255, 0, 0),    # VEHICLE (Red)
    1: (0, 255, 0),    # PEDESTRIAN (Green)
    2: (0, 0, 255),    # BICYCLE (Blue)
    # Add more colors for other classes
}