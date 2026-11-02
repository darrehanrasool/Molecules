def test_imports():
    import src.config, src.utils, src.model, src.retrieval
    assert True

def test_bin_spectrum():
    import numpy as np
    from src.utils import bin_spectrum
    v = bin_spectrum([100.0, 200.0], [1.0, 0.5])
    assert v.shape[0] > 1000 and v.max() <= 1.0

def test_tanimoto_accepts_compact_fingerprint_database():
    import numpy as np
    from src.retrieval import tanimoto_matrix
    db_fps = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.uint8)
    query_fp = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    scores = tanimoto_matrix(query_fp, db_fps)
    assert db_fps.dtype == np.uint8
    np.testing.assert_allclose(scores, [2 / 3, 0.25], rtol=1e-6)


def test_resume_history_requires_contiguous_completed_epochs():
    import pytest
    from src.train import _validate_resume_history

    history = {
        "epoch": [1, 2],
        "train_loss": [0.1, 0.09],
        "val_loss": [0.2, 0.19],
        "val_ap": [0.3, 0.31],
        "val_auc": [0.7, 0.71],
        "val_bit_acc": [0.9, 0.91],
        "lr": [0.001, 0.0009],
    }
    _validate_resume_history(history, 2)

    with pytest.raises(ValueError, match="completed history"):
        _validate_resume_history(history, 3)