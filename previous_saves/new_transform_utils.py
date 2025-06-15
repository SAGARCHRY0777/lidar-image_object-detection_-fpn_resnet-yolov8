# new_transform_utils.py (replacement for argoverse.utils.transform)
import numpy as np

def quat2rotmat(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion into a 3x3 rotation matrix.
    Args:
        q: A 4-element numpy array representing the quaternion (w, x, y, z).
    Returns:
        A 3x3 rotation matrix.
    """
    w, x, y, z = q[0], q[1], q[2], q[3]
    
    # Normalize the quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0:
        return np.eye(3) # Return identity if quaternion is zero
    w /= norm
    x /= norm
    y /= norm
    z /= norm

    rot_mat = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    return rot_mat