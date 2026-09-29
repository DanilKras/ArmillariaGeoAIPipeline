import numpy as np
from scipy.spatial import KDTree


def generate_mock_absences(presences: np.ndarray, buffer_m: float, n_samples: int) -> np.ndarray:
    tree = KDTree(presences)
    min_x, max_x = presences[:, 0].min() - 20000, presences[:, 0].max() + 20000
    min_y, max_y = presences[:, 1].min() - 20000, presences[:, 1].max() + 20000

    candidates = []
    while len(candidates) < n_samples:
        points = np.column_stack([
            np.random.uniform(min_x, max_x, n_samples * 2),
            np.random.uniform(min_y, max_y, n_samples * 2)
        ])
        distances, _ = tree.query(points)
        valid_points = points[distances > buffer_m]
        candidates.extend(valid_points.tolist())

    return np.array(candidates[:n_samples])


def test_pseudo_absences_spatial_buffer():
    np.random.seed(42)
    presences = np.array([
        [4000000.0, 3000000.0],
        [4005000.0, 3002000.0],
        [4010000.0, 3005000.0]
    ])

    buffer_meters = 5000.0
    absences = generate_mock_absences(presences, buffer_m=buffer_meters, n_samples=20)

    tree = KDTree(presences)
    distances, _ = tree.query(absences)

    assert len(absences) == 20
    assert np.all(
        distances > buffer_meters), "Found pseudo-absence points within the buffer distance from presence points."
