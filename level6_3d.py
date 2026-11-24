"""
Level 6 - Kerangka 3D yang bisa diputar dengan mouse.

    python level6_3d.py                         (kamera, tekan Spasi untuk mengambil)
    python level6_3d.py --source foto.jpg

Memakai world landmarks: perkiraan meter dengan pusat di pinggul.
Sumbu z adalah TEBAKAN dari satu gambar 2D, bukan hasil ukur.
"""
import matplotlib

from pose_util import FOLDER, TULANG, ambil_satu, argumen, warna_vis

args = argumen("Level 6: kerangka 3D").parse_args()
if args.no_show:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (setelah backend dipilih)

frame, orang = ambil_satu(args)

# Sumbu MediaPipe: x ke kanan gambar, y ke bawah, z menjauhi kamera.
# Sumbu grafik: (x, z, -y) supaya orangnya berdiri tegak.
X, Y, Z = orang.dunia[:, 0], orang.dunia[:, 2], -orang.dunia[:, 1]
if args.source.isdigit():
    X = -X  # tampilan kamera dicerminkan
fig = plt.figure(figsize=(11, 5.5), layout="constrained")
kiri = fig.add_subplot(1, 2, 1)
kiri.imshow(frame[:, :, ::-1])
for a, b in TULANG:
    kiri.plot(*orang.px[[a, b]].T, color="deepskyblue", lw=2)
kiri.scatter(*orang.px.T, s=14, color="white", edgecolors="black", zorder=3)
kiri.set_xlim(0, frame.shape[1])
kiri.set_ylim(frame.shape[0], 0)
kiri.set_title("2D pada gambar")
kiri.axis("off")

ax = fig.add_subplot(1, 2, 2, projection="3d")
for a, b in TULANG:
    ax.plot(X[[a, b]], Y[[a, b]], Z[[a, b]], color="steelblue", lw=2)
ax.scatter(X, Y, Z, s=25, color=[tuple(c / 255 for c in warna_vis(v)[::-1]) for v in orang.vis], depthshade=False)
ax.set(xlabel="x (m)", ylabel="z: menjauhi kamera (m)", zlabel="atas (m)", title="3D (putar dengan mouse)")
batas = max(abs(v).max() for v in (X, Y, Z))
ax.set(xlim=(-batas, batas), ylim=(-batas, batas), zlim=(-batas, batas))
ax.set_box_aspect((1, 1, 1))
ax.view_init(elev=12, azim=-70)

tujuan = args.save or FOLDER / "hasil" / "level6.png"
(FOLDER / "hasil").mkdir(exist_ok=True)
fig.savefig(tujuan, dpi=110)
print("Tersimpan:", tujuan)
if not args.no_show:
    plt.show()

# COBA:
# 1. Putar grafik ke samping. Apakah posisi tangan depan-belakang masuk akal?
# 2. Ambil dua kali dengan pose yang sama. Seberapa besar z berubah dibanding x dan y?
