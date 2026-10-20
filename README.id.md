# TokoKita: Prediksi Churn Bebas Kebocoran Label dengan Validasi Out-of-Time

*Awalnya dikembangkan sebagai proyek tugas kuliah (AI for Data & Predictive Analytics), dibangun ulang dan dikembangkan lebih lanjut secara mandiri.*

[English](README.md) | Bahasa Indonesia (ringkas)

Proyek ini memprediksi pelanggan **aktif** sebuah marketplace yang **tidak akan memesan sama sekali dalam 30 hari ke depan**, memvalidasi model pada periode masa depan yang belum pernah dilihat, lalu mengubah skor menjadi keputusan retensi yang diuji ketahanannya terhadap asumsinya sendiri.

> **Seluruh data bersifat sintetis**, dibuat oleh simulasi siklus hidup pelanggan. Tidak ada perusahaan, pelanggan, atau transaksi nyata. Kesimpulan berlaku untuk simulasi ini, bukan untuk toko sungguhan.

## Hasil utama (uji out-of-time, data sintetis, seed 42)

* Periode uji: snapshot 1 Nov, 15 Nov, dan 1 Des 2024; semua jendela label data latih dan validasi sudah selesai sebelum 1 Nov.
* Model terbaik menurut validasi: **XGBoost, AUC 0,800** (CI 95% 0,783 sampai 0,817). Logistic Regression 0,788, Random Forest 0,796, LightGBM 0,800. Selisih antarmodel kecil.
* 10% pelanggan berisiko tertinggi benar-benar churn **76%** (tingkat churn dasar 32%), lift 2,4 kali.
* Probabilitas cukup terkalibrasi (ECE 0,033; Brier 0,166).
* **Kebocoran label:** bila fitur dihitung di akhir jendela label (kesalahan yang ditemukan analisis awal), AUC melonjak menjadi **0,98 sampai 0,99**; `recency_days` saja mencapai AUC 1,000. Analisis awal sendiri menemukan masalah ini dan beralih ke evaluasi out-of-time; repositori ini mengubah temuan itu menjadi pagar pengaman: pembagian waktu dengan *purge*, audit AUC satu-fitur otomatis, dan tes pytest.
* **Keputusan bisnis (di bawah asumsi yang dinyatakan):** aturan bernilai-sadar (hubungi bila `p >= biaya / (save_rate x nilai pelanggan)`) menghasilkan sekitar **+5,3 juta IDR per 1.000 pelanggan yang diskor** (CI 95% 2,6 sampai 8,5 juta), tetapi baru impas bila *save rate* sebenarnya di atas sekitar **9,2%**. Karena itu uji acak (holdout) wajib dilakukan sebelum peluncuran. Bila operasional butuh satu ambang tunggal: **p >= 0,79** (sekitar 3% pelanggan teratas), lebih kecil untungnya tetapi jauh lebih kecil rugi bila tawaran kurang efektif.

## Pendekatan singkat

* **Data sintetis:** 3.000 pelanggan, 46.323 pesanan (Jan 2023 sampai Des 2024). Tiap pelanggan punya tingkat nilai laten, laju pesanan, dan tanggal *defection* laten; laju pesanan menurun 20 sampai 60 hari sebelum defection sehingga churn meninggalkan jejak perilaku. Generator deterministik (seed tetap) dan ditulis ulang dari nol.
* **Fitur:** 35 fitur (RFM, perilaku, tempo, momentum, CLV, demografi), seluruhnya dihitung dari pesanan sampai tanggal `as_of`.
* **Label:** `churn_30d = 1` bila tidak ada pesanan pada `(as_of, as_of + 30 hari]`.
* **Pembagian waktu dengan purge:** latih, validasi, uji out-of-time; baris yang jendela labelnya melewati batas periode dibuang.
* **Model:** Logistic Regression, Random Forest, XGBoost, LightGBM; kalibrasi isotonik pada validasi; *bootstrap* tingkat pelanggan (1.000 kali) untuk selang kepercayaan.
* **Net benefit:** dihitung dari hasil nyata pada data uji, dengan asumsi biaya kontak 35.000 IDR, save rate 15%, margin 25%, 6 bulan retensi; diuji sensitivitasnya.

## Keterbatasan

Data sintetis dan satu jendela uji (dua bulan); asumsi ekonomi bukan hasil pengukuran dan save rate tidak dapat diidentifikasi dari data observasional (model churn bukan model uplift); tanpa pencarian hiperparameter; definisi churn tidak dibedakan per segmen.

## Cara menjalankan

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                               # 18 tes
python scripts/run_pipeline.py       # sekitar 4 menit: gambar + reports/results.json
jupyter nbconvert --to notebook --execute --inplace notebooks/tokokita_churn.ipynb
```

Struktur repositori, rincian metodologi, dan tabel lengkap ada di [README.md](README.md).

## Pengungkapan bantuan AI

Analisis awal, perumusan masalah, dan temuannya adalah tugas kuliah saya. Untuk versi portofolio ini saya memakai asisten AI (Claude dari Anthropic) untuk membantu merapikan struktur repositori, menulis generator data sintetis baru, kode paket, tes, dan dokumentasi. Saya mengarahkan pekerjaan ini dan bertanggung jawab atasnya; setiap angka berasal dari hasil menjalankan kode di repositori ini (`reports/results.json`).

## Lisensi

MIT, lihat [LICENSE](LICENSE).
