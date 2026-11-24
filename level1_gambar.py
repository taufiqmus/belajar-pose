"""
Level 1 - Dari satu gambar menjadi 33 titik.

    python level1_gambar.py                     (kamera, tekan Spasi untuk mengambil)
    python level1_gambar.py --source foto.jpg

Keluaran: tabel di terminal, hasil/level1.jpg, dan hasil/landmark.csv
"""
import csv

import cv2

from pose_util import FOLDER, NAMA, ambil_satu, argumen, gambar_kerangka, simpan_gambar

args = argumen("Level 1: landmark dari satu gambar").parse_args()
frame, orang = ambil_satu(args)

print(f"{'no':>2}  {'nama':<24}{'x':>7}{'y':>7}{'z':>7}{'yakin':>7}  catatan")
for i, nama in enumerate(NAMA):
    x, y, z = orang.norm[i]
    catatan = "" if orang.di_dalam[i] else "DI LUAR GAMBAR (tebakan)"
    print(f"{i:>2}  {nama:<24}{x:>7.2f}{y:>7.2f}{z:>7.2f}{orang.vis[i]:>7.2f}  {catatan}")

hasil = FOLDER / "hasil"
hasil.mkdir(exist_ok=True)
with open(hasil / "landmark.csv", "w", newline="", encoding="utf-8") as f:
    tabel = csv.writer(f)
    tabel.writerow(["no", "nama", "x", "y", "z", "visibility"])
    for i, nama in enumerate(NAMA):
        tabel.writerow([i, nama, *(round(float(v), 4) for v in (*orang.norm[i], orang.vis[i]))])

gambar_kerangka(frame, orang, nomor=True)
tujuan = args.save or hasil / "level1.jpg"
simpan_gambar(tujuan, frame)
print("\nTersimpan:", tujuan, "dan", hasil / "landmark.csv")
if not args.no_show:
    cv2.imshow("Level 1 (tekan tombol apa saja)", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# COBA:
# 1. Tutup satu tangan di belakang badan. Titik mana yang nilai yakinnya turun?
# 2. Potong foto sehingga tangan keluar gambar. Apakah model tetap memberi koordinat?
# 3. Ubah x dan y menjadi piksel: x * lebar gambar, y * tinggi gambar. Cocokkan dengan orang.px
#    (pada kamera, tampilan dicerminkan, jadi x menjadi 1 - x).
