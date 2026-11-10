"""Loading and selecting the first entries of a MultiVerse (17-column) crossing table."""
import numpy as np
import pandas as pd
import pytest

import magi

ROW = "{ev}\t{name}\t{E}\t0\t{tid}\t1\tcompt\t10.0\t20.0\t30.0\t0.0\t0.0\t-1.0\t1\t13\t500.0{extra}\n"


def _write(tmp_path, flags, with_flag=True):
    p = tmp_path / ("cross17.dat" if with_flag else "cross16.dat")
    with open(p, "w") as f:
        for i, fl in enumerate(flags):
            f.write(ROW.format(ev=i, name="gamma", E=0.1 + i, tid=i + 2,
                               extra=f"\t{fl}" if with_flag else ""))
    return str(p)


def test_seventeen_columns_are_read_with_the_flag(tmp_path):
    df = magi.load_detector_table(filepath=_write(tmp_path, [0, 1, 0, 0]), sep=r"\s+")
    assert "AncestorInside" in df.columns and df["AncestorInside"].tolist() == [0, 1, 0, 0]
    assert df["PrimaryPDG"].iloc[0] == 13 and df["PrimaryEnergy"].iloc[0] == 500.0


def test_sixteen_columns_are_unchanged(tmp_path):
    df = magi.load_detector_table(filepath=_write(tmp_path, [0, 0], with_flag=False), sep=r"\s+")
    assert "AncestorInside" not in df.columns and len(df) == 2


def test_select_first_entry_keeps_only_unflagged_rows(tmp_path, capsys):
    df = magi.load_detector_table(filepath=_write(tmp_path, [0, 1, 0, 1, 0]), sep=r"\s+")
    out = magi.select_first_entry(df)
    assert len(out) == 3 and "AncestorInside" not in out.columns
    assert out["EventId"].tolist() == [0, 2, 4]
    assert "40.00 % flagged" in capsys.readouterr().out
    assert "AncestorInside" in magi.select_first_entry(df, drop_flag=False).columns


def test_select_first_entry_needs_the_flag(tmp_path):
    df = magi.load_detector_table(filepath=_write(tmp_path, [0, 0], with_flag=False), sep=r"\s+")
    with pytest.raises(ValueError, match="AncestorInside"):
        magi.select_first_entry(df)
