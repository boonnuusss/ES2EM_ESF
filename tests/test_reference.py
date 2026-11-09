from pathlib import Path

from esf_sim.reference import load_figure2_reference, verify_reference_files


def test_sciencedb_v2_integrity_and_shapes():
    root = Path("data/reference")
    integrity = verify_reference_files(root)
    assert all(item["matches"] for item in integrity.values())
    arrays = load_figure2_reference(root / "numDensity.mat")
    assert set(arrays) == {"first_order", "van_leer", "pdm7"}
    assert all(item[0].shape == (401, 201) for item in arrays.values())
