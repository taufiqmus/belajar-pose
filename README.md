# Belajar Pose Estimation dari Kode

Dari satu gambar sampai kerangka 3D, dengan landmark yang terlihat di setiap langkah.
Memakai MediaPipe Pose Landmarker (33 titik tubuh).

## Pasang

Python 3.10 sampai 3.12 disarankan.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python test_pose.py
```

Model `lite` sudah ada di folder `models`, jadi setelah instalasi tidak butuh internet.
Model `full` dan `heavy` diunduh sekali saat pertama dipilih.

## Urutan belajar

| Level | Berkas | Yang dipelajari | Yang terlihat |
|---|---|---|---|
| 1 | `level1_gambar.py` | koordinat, z, dan visibility 33 titik | titik bernomor, tabel, CSV |
| 2 | `level2_webcam.py` | deteksi real-time dan keyakinan model | kerangka berwarna, FPS |
| 3 | `level3_sudut.py` | sudut sendi dari tiga titik | busur dan angka sudut |
| 4 | `level4_repetisi.py` | sudut menjadi penghitung gerakan | jumlah curl atau squat |
| 5 | `level5_filter.py` | derau, EMA, dan One Euro Filter | jejak titik dan grafik |
| 6 | `level6_3d.py` | world landmarks | kerangka 3D yang bisa diputar |

Setiap berkas bisa dijalankan sendiri, misalnya `python level2_webcam.py`.
Di akhir tiap berkas ada bagian **COBA** untuk latihan. `pose_util.py` berisi bagian bersama.

## Opsi

| Opsi | Fungsi |
|---|---|
| `--source` | indeks kamera (0, 1, ..), path video, atau path gambar |
| `--model` | `lite`, `full`, atau `heavy` |
| `--backend` | `dshow` atau `msmf` jika kamera Windows tidak terbuka |
| `--save` | simpan hasil, mis. `hasil/rekaman.mp4` atau `hasil/foto.jpg` |
| `--no-show` | tanpa jendela |
| `--max-frames` | berhenti setelah N frame |
| `--gerakan`, `--sisi` | khusus level 4: `curl`/`squat`, `kiri`/`kanan` |
| `--titik` | khusus level 5: nomor titik 0 sampai 32 |

Tombol umum: `q` atau `Esc` keluar. Tombol lain tertulis di awal tiap berkas.

## Empat jebakan

1. **RGB dan BGR.** MediaPipe memakai RGB, OpenCV memakai BGR.
2. **Cermin.** Deteksi dulu, baru cerminkan. Kalau terbalik, kiri dan kanan tertukar.
3. **z bukan hasil ukur.** Model menebaknya dari satu gambar.
4. **Bagian tubuh di luar gambar tetap ditebak**, kadang di tempat yang salah dengan visibility tinggi.

## Masalah umum

- `module 'mediapipe' has no attribute 'solutions'`: MediaPipe versi baru sudah menghapus
  `mp.solutions`. Repo ini memakai Tasks API, jadi tidak terpengaruh.
- Kamera tidak terbuka: tutup Zoom atau browser, coba `--source 1` atau `--backend dshow`.
- `cv2` error setelah memasang `opencv-python`: hapus keduanya, lalu pasang ulang.
  `pip uninstall -y opencv-python opencv-contrib-python` lalu `pip install opencv-contrib-python`

## Lisensi model

Model Pose Landmarker berasal dari Google MediaPipe. Kartu model BlazePose GHUM 3D mencantumkan Apache License 2.0.
