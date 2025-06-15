# new_se3.py (replacement for argoverse.utils.se3)
import numpy as np

class SE3:
    """Represents a 3D rigid body transformation (rotation and translation)."""

    def __init__(self, rotation: np.ndarray = np.eye(3), translation: np.ndarray = np.zeros(3)) -> None:
        """
        Args:
            rotation: A 3x3 rotation matrix.
            translation: A 3-element translation vector.
        """
        if rotation.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3.")
        if translation.shape != (3,):
            raise ValueError("Translation vector must be 3-element.")

        self.rotation = rotation
        self.translation = translation

    @property
    def transform_matrix(self) -> np.ndarray:
        """Get the 4x4 homogeneous transformation matrix."""
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T

    def inverse(self) -> 'SE3':
        """Compute the inverse transformation."""
        inv_rotation = self.rotation.T
        inv_translation = -inv_rotation @ self.translation
        return SE3(rotation=inv_rotation, translation=inv_translation)

    def transform_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """
        Transform a point cloud.
        Args:
            points: Nx3 array of points.
        Returns:
            Nx3 array of transformed points.
        """
        if points.shape[1] != 3:
            raise ValueError("Points must be Nx3.")
        
        return (self.rotation @ points.T + self.translation[:, np.newaxis]).T