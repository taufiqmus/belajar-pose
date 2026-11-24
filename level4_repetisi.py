"""
Level 4 - Dari sudut menjadi penghitung gerakan.

    python level4_repetisi.py                    (curl: menekuk siku)
    python level4_repetisi.py --gerakan squat    (seluruh kaki harus terlihat)

Dua ambang (histeresis) dipakai supaya sudut yang bergetar di sekitar satu ambang
tidak terhitung berkali-kali. Tombol: r ulang dari nol, q keluar
"""
import cv2

from pose_util import Sesi, argumen, gambar_kerangka, sudut, tulis

GERAKAN = {  # (titik sisi kiri, titik sisi kanan, ambang tekuk, ambang lurus)
    "curl": ((11, 13, 15), (12, 14, 16), 60, 150),
    "squat": ((23, 25, 27), (24, 26, 28), 100, 160),
}


class Penghitung:
    """Satu repetisi = menekuk sampai < tekuk, lalu kembali lurus sampai > lurus."""

    def __init__(self, tekuk, lurus):
        self.tekuk, self.lurus, self.sudah_tekuk, self.jumlah = tekuk, lurus, False, 0

    def perbarui(self, derajat):
        if derajat < self.tekuk:
            self.sudah_tekuk = True
        elif derajat > self.lurus and self.sudah_tekuk:
            self.sudah_tekuk = False
            self.jumlah += 1
        return self.jumlah


if __name__ == "__main__":
    ap = argumen("Level 4: penghitung repetisi")
    ap.add_argument("--gerakan", choices=GERAKAN, default="curl")
    ap.add_argument("--sisi", choices=["kiri", "kanan"], default="kanan", help="lengan/kaki yang dihitung")
    args = ap.parse_args()
    kiri, kanan, tekuk, lurus = GERAKAN[args.gerakan]
    sisi = list(kiri if args.sisi == "kiri" else kanan)
    hitung = Penghitung(tekuk, lurus)
    with Sesi(args) as s:
        for frame, orang in s:
            if orang is not None:
                gambar_kerangka(frame, orang)
            if orang is not None and orang.yakin(*sisi):
                d = sudut(*orang.px[sisi])
                hitung.perbarui(d)
                # batang: penuh = lurus, kosong = tekuk
                isi = min(max((d - tekuk) / (lurus - tekuk), 0), 1) if lurus > tekuk else 0
                cv2.rectangle(frame, (15, 130), (45, 330), (60, 60, 60), -1)
                cv2.rectangle(frame, (15, int(330 - 200 * isi)), (45, 330), (0, 220, 255), -1)
                tulis(frame, f"{d:.0f}", (55, 330))
            else:
                tulis(frame, f"Sendi {args.sisi} belum terlihat jelas", (15, 110), 0.6, (80, 80, 255))
            tulis(frame, f"{args.gerakan}: {hitung.jumlah}", (15, 65), 1.5, (0, 255, 120))
            s.tampil(frame)
            if s.tombol == "r":
                hitung = Penghitung(tekuk, lurus)
    print("Total repetisi:", hitung.jumlah)

# COBA:
# 1. Ubah ambang menjadi sama (misalnya 100 dan 100). Apa yang terjadi pada hitungan?
# 2. Tambahkan gerakan baru: angkat lengan ke samping (titik 13-11-23).
# 3. Beri pesan kalau gerakan terlalu cepat (pakai s.waktu).
