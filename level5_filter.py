"""
Level 5 - Kenapa titik bergetar, dan cara menghaluskannya.

    python level5_filter.py                      (gerakkan pergelangan tangan kanan)
    python level5_filter.py --titik 0            (0 = hidung; lihat nomor di level 1)

Merah = mentah, hijau = EMA, kuning = One Euro.
Tombol: + dan - mengubah alpha EMA, q keluar
"""
from collections import deque

import cv2
import numpy as np

from pose_util import NAMA, Sesi, argumen, tulis

WARNA = {"mentah": (80, 80, 255), "EMA": (80, 220, 80), "One Euro": (0, 220, 255)}


class EMA:
    """Rata-rata eksponensial: alpha kecil = halus tetapi tertinggal."""

    def __init__(self, alpha):
        self.alpha, self.x = alpha, None

    def __call__(self, x, t):
        self.x = x if self.x is None else self.x + self.alpha * (x - self.x)
        return self.x


class OneEuro:
    """Halus saat pelan, cepat tanggap saat gerakan cepat (Casiez dkk., 2012)."""

    def __init__(self, min_cutoff=1.0, beta=3.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.t = None
        self.dx = 0.0

    @staticmethod
    def _alpha(cutoff, dt):
        return 1 / (1 + 1 / (2 * np.pi * cutoff * dt))

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.t = x, t
            return x
        dt = max(t - self.t, 1e-3)
        self.t = t
        self.dx += self._alpha(self.d_cutoff, dt) * ((x - self.x) / dt - self.dx)
        self.x += self._alpha(self.min_cutoff + self.beta * abs(self.dx), dt) * (x - self.x)
        return self.x


def gambar_grafik(lebar, jejak, tinggi=160):
    grafik = np.full((tinggi, lebar, 3), 30, np.uint8)
    if len(jejak["mentah"]) > 1:
        semua = np.concatenate([[p[0] for p in d] for d in jejak.values()])
        rentang = max(semua.max() - semua.min(), 0.02)  # jangan memperbesar derau sampai tampak seperti gerakan
        tengah = (semua.max() + semua.min()) / 2
        for nama, d in jejak.items():
            titik = [(int(i * lebar / d.maxlen), int(tinggi * 0.55 - (p[0] - tengah) / rentang * tinggi * 0.7))
                     for i, p in enumerate(d)]
            cv2.polylines(grafik, [np.array(titik, np.int32)], False, WARNA[nama], 2, cv2.LINE_AA)
    for i, nama in enumerate(WARNA):
        tulis(grafik, nama, (10 + i * 100, 20), 0.5, WARNA[nama])
    tulis(grafik, "posisi x terhadap waktu", (330, 20), 0.5)
    return grafik


if __name__ == "__main__":
    ap = argumen("Level 5: menghaluskan landmark")
    ap.add_argument("--titik", type=int, default=16, choices=range(33), metavar="0..32")
    args = ap.parse_args()
    alpha = 0.3
    saring = {"EMA": (EMA(alpha), EMA(alpha)), "One Euro": (OneEuro(), OneEuro())}
    jejak = {nama: deque(maxlen=90) for nama in WARNA}
    with Sesi(args) as s:
        for frame, orang in s:
            h, w = frame.shape[:2]
            if orang is not None and orang.yakin(args.titik):
                p = orang.px[args.titik] / (w, h)
                jejak["mentah"].append(p)
                for nama, (fx, fy) in saring.items():
                    jejak[nama].append(np.array([fx(p[0], s.waktu), fy(p[1], s.waktu)]))
            else:
                tulis(frame, "Titik tidak terlihat", (15, 95), 0.6, (80, 80, 255))
            for nama, d in jejak.items():
                if d:
                    titik = (np.array(d)[-30:] * (w, h)).astype(np.int32)
                    cv2.polylines(frame, [titik], False, WARNA[nama], 2, cv2.LINE_AA)
                    cv2.circle(frame, tuple(titik[-1].tolist()), 8, WARNA[nama], -1, cv2.LINE_AA)
            tulis(frame, f"{NAMA[args.titik]}   alpha EMA = {alpha:.2f}", (15, 30))
            tulis(frame, f"{s.fps:.0f} FPS", (15, 62), 0.6)
            s.tampil(np.vstack([frame, gambar_grafik(w, jejak)]))
            if s.tombol in ("+", "=", "-"):
                alpha = float(np.clip(alpha + (-0.05 if s.tombol == "-" else 0.05), 0.05, 1))
                for f in saring["EMA"]:
                    f.alpha = alpha

# COBA:
# 1. Diamkan tangan. Garis mana yang paling bergetar?
# 2. Gerakkan tangan cepat. Garis mana yang paling tertinggal?
# 3. Cari alpha EMA yang pas untuk keduanya. Apakah ada? Bandingkan dengan One Euro.
