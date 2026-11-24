import numpy as np
from eval.round8_clustering import incident_vectors, silhouette_kmeans


def test_predicted_groups_keep_background_and_mixed_truth():
    rows = [
        dict(incident_id="a", anomaly_type="one"),
        dict(incident_id="b", anomaly_type="two"),
        dict(incident_id=None),
    ]
    matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)
    x, y, groups = incident_vectors(rows, np.eye(3), matrix)
    assert set(y) == {"mixed", "background"}
    assert len(x) == 2


def test_silhouette_k_is_selected_without_truth():
    x = np.array([[0.0, 0.0], [0, 0.1], [0.1, 0], [10, 10], [10, 10.1], [10.1, 10]])
    assert len(set(silhouette_kmeans(x))) == 2
