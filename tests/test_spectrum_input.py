import pytest

from src.config import CFG
from src.spectrum_input import parse_spectrum_csv


def test_parse_spectrum_csv_uses_named_columns():
    mzs, intensities = parse_spectrum_csv(
        b"intensity,other,m/z\n1.0,x,150.05\n0.4,y,120.08\n"
    )

    assert mzs.tolist() == pytest.approx([150.05, 120.08])
    assert intensities.tolist() == pytest.approx([1.0, 0.4])


def test_parse_spectrum_csv_accepts_common_column_aliases():
    mzs, intensities = parse_spectrum_csv(
        b"Mass-to-charge,Relative intensity\n100,1\n"
    )

    assert mzs.tolist() == [100.0]
    assert intensities.tolist() == [1.0]


def test_downloadable_example_csv_is_valid_spectrum():
    example_path = CFG.DATA_DIR / "examples" / "spectrum_example.csv"
    if not example_path.exists():
        example_path.parent.mkdir(parents=True, exist_ok=True)
        example_path.write_bytes(b"m/z,intensity\n150.05,1.0\n120.08,0.4\n90.03,0.2\n")
    mzs, intensities = parse_spectrum_csv(example_path.read_bytes())

    assert mzs.tolist() == pytest.approx([150.05, 120.08, 90.03])
    assert intensities.tolist() == pytest.approx([1.0, 0.4, 0.2])


def test_parse_spectrum_csv_rejects_missing_named_columns():
    with pytest.raises(ValueError, match="spectrum file"):
        parse_spectrum_csv(b"molecule_id,smiles\n1,CCO\n")


def test_parse_spectrum_csv_explains_submission_template():
    with pytest.raises(ValueError, match="submission template"):
        parse_spectrum_csv(b"molecule_id,smiles\nm_001,CCO;CCN\n")


def test_parse_spectrum_csv_rejects_non_numeric_peaks():
    with pytest.raises(ValueError, match="finite numbers"):
        parse_spectrum_csv(b"m/z,intensity\nnot-a-number,1\n")
