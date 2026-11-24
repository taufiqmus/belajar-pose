"""
Level 3 - Dari titik menjadi sudut sendi.

    python level3_sudut.py

Sudut di titik B dari tiga titik A-B-C:  cos(sudut) = (BA . BC) / (|BA| |BC|)
Pakai PIKSEL. x dan y ternormalisasi punya skala berbeda kalau gambar tidak persegi,
sehingga sudutnya salah.
"""
import cv2
import numpy as np

from pose_util import Sesi, argumen, gambar_kerangka, sudut, tulis

SENDI = {  # nama: (A, B, C), sudut diukur di B
    "siku kiri": (11, 13, 15), "siku kanan": (12, 14, 16),
    "bahu kiri": (13, 11, 23), "bahu kanan": (14, 12, 24),
    "lutut kiri": (23, 25, 27), "lutut kanan": (24, 26, 28),
}


def gambar_busur(frame, a, b, c):
    """Busur kecil di B, dari arah A ke arah C lewat sisi yang lebih kecil."""
    s1 = np.degrees(np.arctan2(a[1] - b[1], a[0] - b[0]))
    s2 = np.degrees(np.arctan2(c[1] - b[1], c[0] - b[0]))
    beda = (s2 - s1 + 180) % 360 - 180
    pusat = (int(b[0]), int(b[1]))
    cv2.ellipse(frame, pusat, (28, 28), 0, float(s1), float(s1 + beda), (0, 220, 255), 3, cv2.LINE_AA)


if __name__ == "__main__":
    args = argumen("Level 3: sudut sendi").parse_args()
    with Sesi(args) as s:
        for frame, orang in s:
            if orang is not None:
                gambar_kerangka(frame, orang)
                for a, b, c in SENDI.values():
                    if not orang.yakin(a, b, c):  # jangan menghitung dari tebakan
                        continue
                    A, B, C = orang.px[[a, b, c]]
                    gambar_busur(frame, A, B, C)
                    tulis(frame, f"{sudut(A, B, C):.0f}", B + (14, -14), 0.7, (0, 220, 255))
            tulis(frame, f"{s.fps:.0f} FPS", (15, 30))
            s.tampil(frame)

# COBA:
# 1. Luruskan siku, lalu tekuk 90 derajat. Berapa galat sudut yang terbaca?
# 2. Arahkan lengan ke kamera sambil siku tetap lurus. Kenapa sudutnya berubah?
# 3. Hitung sudut yang sama dari orang.dunia (3D). Mana yang lebih dekat ke sudut asli?
