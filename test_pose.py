"""Uji logika tanpa kamera:  python test_pose.py"""
import numpy as np

from level4_repetisi import Penghitung
from level5_filter import EMA, OneEuro
from pose_util import NAMA, TULANG, sudut


def test_sudut():
    assert abs(sudut([1, 0], [0, 0], [0, 1]) - 90) < 1e-6
    assert abs(sudut([1, 0], [0, 0], [-1, 0]) - 180) < 1e-6
    assert np.isnan(sudut([0, 0], [0, 0], [1, 1]))


def test_penghitung_tahan_getar():
    p = Penghitung(60, 150)
    for d in [170, 100, 58, 62, 57, 63, 149, 151, 148, 152, 170, 40, 160]:
        p.perbarui(d)
    assert p.jumlah == 2


def test_filter():
    rng = np.random.default_rng(0)
    t = np.arange(120) / 30
    diam = 0.5 + rng.normal(0, 0.01, t.size)
    for f in (EMA(0.3), OneEuro()):
        hasil = [f(x, ti) for x, ti in zip(diam, t)]
        assert np.std(hasil[30:]) < np.std(diam[30:])  # lebih tenang dari sinyal mentah


def test_data():
    assert len(NAMA) == len(set(NAMA)) == 33
    assert len(TULANG) == 35 and all(0 <= a < 33 and 0 <= b < 33 for a, b in TULANG)


if __name__ == "__main__":
    for nama, fungsi in list(globals().items()):
        if nama.startswith("test_"):
            fungsi()
            print("OK", nama)
