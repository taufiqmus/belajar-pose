"""
pose_util.py - bagian yang dipakai bersama semua level.

Alur: sumber (kamera/video/gambar) -> PoseLandmarker -> Orang -> gambar -> tampil/simpan
Baca berkas ini setelah level1 dan level2 selesai dicoba.
"""
import argparse
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

FOLDER = Path(__file__).parent
URL_MODEL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{0}/float16/1/pose_landmarker_{0}.task")
EKSTENSI_GAMBAR = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 33 titik. "kiri/kanan" = kiri/kanan ORANGNYA, bukan sisi layar.
NAMA = [
    "hidung", "mata kiri dalam", "mata kiri", "mata kiri luar", "mata kanan dalam",
    "mata kanan", "mata kanan luar", "telinga kiri", "telinga kanan", "mulut kiri",
    "mulut kanan", "bahu kiri", "bahu kanan", "siku kiri", "siku kanan",
    "pergelangan kiri", "pergelangan kanan", "kelingking kiri", "kelingking kanan",
    "telunjuk kiri", "telunjuk kanan", "jempol kiri", "jempol kanan", "pinggul kiri",
    "pinggul kanan", "lutut kiri", "lutut kanan", "pergelangan kaki kiri",
    "pergelangan kaki kanan", "tumit kiri", "tumit kanan", "ujung kaki kiri",
    "ujung kaki kanan",
]

# Pasangan titik yang dihubungkan garis (kerangka).
TULANG = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


class Orang:
    """Hasil deteksi satu orang. Baris ke-i = titik ke-i pada NAMA."""

    def __init__(self, hasil, lebar, tinggi, cermin):
        lm, dunia = hasil.pose_landmarks[0], hasil.pose_world_landmarks[0]
        self.norm = np.array([[p.x, p.y, p.z] for p in lm])      # x, y: 0..1 dari ukuran gambar
        self.vis = np.array([p.visibility for p in lm])          # keyakinan model, 0..1
        self.dunia = np.array([[p.x, p.y, p.z] for p in dunia])  # perkiraan meter, pusat di pinggul
        x = 1 - self.norm[:, 0] if cermin else self.norm[:, 0]
        self.px = np.stack([x * lebar, self.norm[:, 1] * tinggi], axis=1)  # piksel pada tampilan
        # Model tetap menebak titik di luar gambar, kadang dengan visibility tinggi.
        self.di_dalam = np.all((self.norm[:, :2] >= 0) & (self.norm[:, :2] <= 1), axis=1)

    def yakin(self, *titik, batas=0.5):
        return all(self.vis[i] >= batas and self.di_dalam[i] for i in titik)


def sudut(a, b, c):
    """Sudut di titik b (derajat, 0..180), dibentuk garis b-a dan b-c."""
    v1, v2 = np.asarray(a, float) - b, np.asarray(c, float) - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n == 0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1, 1))))


def warna_vis(v):
    """Merah (tidak yakin) ke hijau (yakin), format BGR."""
    v = float(np.clip(v, 0, 1))
    return 0, int(255 * v), int(255 * (1 - v))


def gambar_titik(frame, orang, nomor=False):
    for i, p in enumerate(orang.px.astype(int).tolist()):
        cv2.circle(frame, tuple(p), 5, warna_vis(orang.vis[i]), -1, cv2.LINE_AA)
        cv2.circle(frame, tuple(p), 5, (20, 20, 20), 1, cv2.LINE_AA)
        if nomor:
            tulis(frame, str(i), (p[0] + 6, p[1] - 6), 0.45)


def gambar_kerangka(frame, orang, nomor=False):
    px = orang.px.astype(int).tolist()
    for a, b in TULANG:
        cv2.line(frame, tuple(px[a]), tuple(px[b]), (255, 200, 80), 2, cv2.LINE_AA)
    gambar_titik(frame, orang, nomor)


def tulis(frame, teks, posisi, skala=0.7, warna=(255, 255, 255)):
    x, y = int(posisi[0]), int(posisi[1])
    # Bayangan digeser, bukan ditebalkan: di OpenCV 5 tebal huruf ikut mengubah lebar teks.
    for geser in ((-1, -1), (1, -1), (-1, 1), (1, 1), (2, 2)):
        cv2.putText(frame, teks, (x + geser[0], y + geser[1]), cv2.FONT_HERSHEY_SIMPLEX, skala, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, teks, (x, y), cv2.FONT_HERSHEY_SIMPLEX, skala, warna, 1, cv2.LINE_AA)


def argumen(deskripsi):
    ap = argparse.ArgumentParser(description=deskripsi)
    ap.add_argument("--source", default="0", help="indeks kamera (0,1,..), path video, atau path gambar")
    ap.add_argument("--model", choices=["lite", "full", "heavy"], default="lite",
                    help="lite paling cepat, heavy paling teliti (full/heavy diunduh sekali)")
    ap.add_argument("--backend", choices=["auto", "dshow", "msmf"], default="auto",
                    help="coba dshow jika kamera Windows tidak terbuka")
    ap.add_argument("--save", help="simpan hasil (mp4 untuk video/kamera, jpg/png untuk gambar)")
    ap.add_argument("--no-show", action="store_true", help="tanpa jendela")
    ap.add_argument("--max-frames", type=int, default=0, help="berhenti setelah N frame (0 = tanpa batas)")
    return ap


def isi_model(jenis):
    path = FOLDER / "models" / f"pose_landmarker_{jenis}.task"
    if not path.exists():
        print(f"Mengunduh model {jenis} ...")
        sementara = path.with_suffix(".unduh")
        try:
            path.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(URL_MODEL.format(jenis), sementara)
            sementara.replace(path)
        except OSError as e:
            sementara.unlink(missing_ok=True)
            sys.exit(f"Model {jenis} gagal diunduh ({e}). Cek internet, atau pakai --model lite.")
    return path.read_bytes()  # dikirim sebagai isi, jadi aman untuk folder dengan spasi


class Sesi:
    """
    Pemakaian:
        with Sesi(args) as s:
            for frame, orang in s:      # orang = None kalau tidak ada yang terdeteksi
                ...menggambar di frame...
                s.tampil(frame)         # s.tombol berisi tombol yang ditekan
    """

    def __init__(self, args):
        self.args, self.tombol, self.n, self._berhenti, self._video = args, "", 0, False, None
        self.gambar = self.cap = None
        self.kamera = args.source.isdigit()
        self.cermin = self.kamera  # kamera dicerminkan, gambar dan video tidak
        if not self.kamera and not Path(args.source).is_file():
            sys.exit(f"File tidak ditemukan: {args.source}")
        if Path(args.source).suffix.lower() in EKSTENSI_GAMBAR:
            self.gambar = cv2.imdecode(np.fromfile(args.source, np.uint8), cv2.IMREAD_COLOR)
            if self.gambar is None:
                sys.exit(f"Gambar tidak bisa dibaca: {args.source}")
        elif self.kamera:
            backend = {"auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}[args.backend]
            self.cap = cv2.VideoCapture(int(args.source), backend)
        else:
            self.cap = cv2.VideoCapture(args.source)
        if self.cap is not None and not self.cap.isOpened():
            sys.exit("Sumber tidak bisa dibuka. Tutup aplikasi lain yang memakai kamera, "
                     "coba --source 1, atau --backend dshow.")
        self.fps_sumber = (self.cap.get(cv2.CAP_PROP_FPS) if self.cap is not None else 0) or 30
        mode = vision.RunningMode.IMAGE if self.gambar is not None else vision.RunningMode.VIDEO
        self.model = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=isi_model(args.model)), running_mode=mode))
        self._t0 = self._t_lalu = time.perf_counter()
        self._ts = -1
        self.fps = 0.0
        self.waktu = 0.0  # detik sejak mulai (video: waktu di dalam video)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.model.close()
        if self.cap is not None:
            self.cap.release()
        if self._video is not None:
            self._video.release()
        if not self.args.no_show:
            cv2.destroyAllWindows()

    def _baca(self):
        if self.gambar is not None:
            return self.gambar.copy() if self.n == 0 else None
        ok, frame = self.cap.read()
        if not ok and self.n == 0:
            sys.exit("Sumber terbuka, tetapi tidak mengirim gambar.")
        return frame if ok else None

    def __iter__(self):
        while not self._berhenti and not (self.args.max_frames and self.n >= self.args.max_frames):
            frame = self._baca()
            if frame is None:
                return
            # MediaPipe memakai RGB, OpenCV memakai BGR.
            citra = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if self.gambar is not None:
                hasil = self.model.detect(citra)
            else:
                self.waktu = time.perf_counter() - self._t0 if self.kamera else self.n / self.fps_sumber
                self._ts = max(self._ts + 1, int(self.waktu * 1000))  # wajib selalu naik
                hasil = self.model.detect_for_video(citra, self._ts)
            # Deteksi memakai frame ASLI. Kalau dicerminkan dulu, kiri dan kanan tertukar.
            if self.cermin:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            self.n += 1
            sekarang = time.perf_counter()
            self.fps += (1 / max(sekarang - self._t_lalu, 1e-6) - self.fps) * 0.1
            self._t_lalu = sekarang
            yield frame, Orang(hasil, w, h, self.cermin) if hasil.pose_landmarks else None

    def tampil(self, frame, judul="Belajar Pose"):
        if self.args.save:
            self._simpan(frame)
        self.tombol = ""
        if self.args.no_show:
            return
        cv2.imshow(judul, frame)
        k = cv2.waitKey(0 if self.gambar is not None else 1) & 0xFF
        self.tombol = chr(k) if k != 255 else ""
        if k in (ord("q"), 27) or cv2.getWindowProperty(judul, cv2.WND_PROP_VISIBLE) < 1:
            self._berhenti = True

    def _simpan(self, frame):
        if self.gambar is not None:
            simpan_gambar(self.args.save, frame)
            return
        if self._video is None:
            h, w = frame.shape[:2]
            Path(self.args.save).parent.mkdir(parents=True, exist_ok=True)
            self._video = cv2.VideoWriter(self.args.save, cv2.VideoWriter_fourcc(*"mp4v"), self.fps_sumber, (w, h))
            if not self._video.isOpened():
                sys.exit(f"Tidak bisa menulis {self.args.save}. Pakai nama berakhiran .mp4.")
        self._video.write(frame)


def simpan_gambar(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix or ".jpg", frame)
    if not ok:
        sys.exit(f"Format tidak didukung: {path}")
    data.tofile(str(path))


def ambil_satu(args):
    """Satu frame: langsung dari gambar, atau tekan Spasi saat kamera/video berjalan."""
    simpan, args.save = args.save, None
    try:
        with Sesi(args) as s:
            for frame, orang in s:
                if orang is not None and (s.gambar is not None or args.no_show):
                    return frame, orang
                if s.gambar is not None:
                    break
                pratinjau = frame.copy()
                if orang is not None:
                    gambar_kerangka(pratinjau, orang)
                tulis(pratinjau, "Spasi: ambil   q: keluar", (15, 30))
                s.tampil(pratinjau)
                if s.tombol == " " and orang is not None:
                    return frame, orang
    finally:
        args.save = simpan
    sys.exit("Tidak ada orang yang diambil. Pastikan kepala dan bahu terlihat.")
