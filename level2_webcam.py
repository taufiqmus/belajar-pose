"""
Level 2 - Landmark langsung dari kamera.

    python level2_webcam.py
    python level2_webcam.py --model full        (bandingkan FPS dan kestabilannya)

Warna titik: hijau = model yakin, merah = tidak yakin.
Tombol: n nomor titik, k kerangka, h latar hitam, s simpan, q keluar
"""
import numpy as np

from pose_util import (FOLDER, NAMA, Sesi, argumen, gambar_kerangka, gambar_titik,
                       simpan_gambar, tulis)

args = argumen("Level 2: landmark real-time").parse_args()
nomor, kerangka, hitam, foto = False, True, False, 0

with Sesi(args) as s:
    for frame, orang in s:
        if hitam:
            frame = np.zeros_like(frame)  # yang tersisa hanya informasi dari model
        if orang is None:
            tulis(frame, "Tidak ada orang", (15, 60), warna=(80, 80, 255))
        else:
            (gambar_kerangka if kerangka else gambar_titik)(frame, orang, nomor)
            i = int(orang.vis.argmin())
            tulis(frame, f"Paling tidak yakin: {NAMA[i]} ({orang.vis[i]:.2f})", (15, 60), 0.55)
        tulis(frame, f"{args.model}  {s.fps:.0f} FPS", (15, 30))
        s.tampil(frame)
        if s.tombol == "n":
            nomor = not nomor
        elif s.tombol == "k":
            kerangka = not kerangka
        elif s.tombol == "h":
            hitam = not hitam
        elif s.tombol == "s":
            foto += 1
            simpan_gambar(FOLDER / "hasil" / f"level2_{foto}.jpg", frame)

# COBA:
# 1. Tekan h. Gerakan apa yang masih bisa dikenali hanya dari titik?
# 2. Bandingkan FPS lite, full, dan heavy. Catat juga mana yang paling tidak bergetar.
# 3. Duduk sampai pinggul tidak terlihat. Titik mana yang berubah merah?
