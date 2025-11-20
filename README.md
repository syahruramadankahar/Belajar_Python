# Analisis Kode Program Facenet

## 1. utils_facenet.py
  Kode Program:
  ```python
  #utils_facenet.py
  import torch, numpy as np, cv2
  from PIL import Image
  from facenet_pytorch import MTCNN, InceptionResnetV1
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  #Detector & aligner
  mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device=device)
  #Embedder (512-dim)
  embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)
  def read_img_bgr(path):
      img = cv2.imread(path)              # BGR
      if img is None:
          raise ValueError(f"Gagal baca: {path}")
      return img
  def bgr_to_pil(img_bgr):
      return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
  @torch.no_grad()
  def face_align(img_bgr):
      """Return aligned face as PIL.Image (160x160) or None if not found."""
      pil = bgr_to_pil(img_bgr)
      aligned = mtcnn(pil)  # tensor [3,160,160] or None
      return aligned
  @torch.no_grad()
  def embed_face_tensor(face_tensor):
      """face_tensor: torch.Tensor [3,160,160] in range [0,1] (from MTCNN)"""
      if face_tensor is None:
          return None
      face_tensor = face_tensor.unsqueeze(0).to(device)  # [1,3,160,160]
      emb = embedder(face_tensor)                        # [1,512]
      return emb.squeeze(0).cpu().numpy()                # (512,)
  @torch.no_grad()
  def embed_from_path(path):
      img = read_img_bgr(path)
      face = face_align(img)
      if face is None:
          return None
      return embed_face_tensor(face)
  def cosine_similarity(a, b, eps=1e-8):
      a = a / (np.linalg.norm(a) + eps)
      b = b / (np.linalg.norm(b) + eps)
      return float(np.dot(a, b))
  ```

  Analisis :
  Kode utils_facenet.py ini berfungsi untuk melakukan proses deteksi wajah, alignment (penyelarasan), dan ekstraksi embedding wajah menggunakan model FaceNet dari pustaka facenet_pytorch. Modul ini memuat pipeline lengkap mulai dari membaca gambar dengan OpenCV, mengonversinya ke format PIL, mendeteksi wajah menggunakan MTCNN, hingga menghasilkan vektor embedding 512-dimensi melalui model InceptionResnetV1. Fungsi embed_from_path() memungkinkan proses embedding dilakukan langsung dari path file gambar, sementara fungsi cosine_similarity() digunakan untuk menghitung tingkat kemiripan antara dua embedding wajah. 

2. verify_pair.py
   Kode Program:
  ```python
  #verify_pair.py
  from utils_facenet import embed_from_path, cosine_similarity
  
  img1 = "data/train/Adam/a1.jpg"
  img2 = "data/train/Isna/i1.jpg"  # ganti sesuai data Anda
  
  emb1 = embed_from_path(img1)
  emb2 = embed_from_path(img2)
  
  if emb1 is None or emb2 is None:
      print("Wajah tidak terdeteksi pada salah satu gambar.")
  else:
      sim = cosine_similarity(emb1, emb2)
      print("Cosine similarity:", sim)
  
  #Threshold umum (awal): 0.8–0.9 (semakin tinggi = semakin ketat)
  threshold = 0.85
  print("Match?", "YA" if sim >= threshold else "TIDAK")
  ```

  Analisis:
  Kode verify_pair.py berfungsi untuk membandingkan dua gambar wajah menggunakan FaceNet dengan cara memanggil fungsi embed_from_path() untuk menghasilkan embedding 512-dimensi dari masing-masing gambar, lalu menghitung tingkat kemiripannya menggunakan cosine_similarity(). Jika salah satu gambar tidak terdeteksi wajahnya, program langsung menampilkan pesan error. Jika keduanya berhasil, nilai kemiripan (cosine similarity) dicetak dan dibandingkan dengan threshold 0.85 untuk menentukan apakah kedua wajah dianggap cocok atau tidak, di mana nilai yang lebih tinggi menandakan wajah yang lebih mirip. Dengan demikian, script ini bekerja sebagai modul verifikasi wajah sederhana berbasis FaceNet.

3. build_embeddings.py
  Kode Program:
  ```python
  # build_embeddings.py
  import os, glob, numpy as np
  from tqdm import tqdm
  from utils_facenet import embed_from_path
  
  def iter_images(root):
      # root = data/train
      classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
      for cls in classes:
          for p in glob.glob(os.path.join(root, cls, "*")):
              yield p, cls
  
  def build_matrix(root):
      X, y, bad = [], [], []
      for path, cls in tqdm(list(iter_images(root))):
          emb = embed_from_path(path)
          if emb is None:
              bad.append(path)
              continue
          X.append(emb); y.append(cls)
      return np.array(X), np.array(y), bad
  
  if __name__ == "__main__":
      X, y, bad = build_matrix("data/train")
      print("Embeddings:", X.shape, "Labels:", y.shape, "Gagal deteksi:", len(bad))
      np.save("X_train.npy", X)
      np.save("y_train.npy", y)
  ```

  Analisis:
  Kode build_embeddings.py digunakan untuk menghasilkan embedding wajah dari seluruh gambar dalam folder dataset, lalu menyimpannya sebagai file NumPy untuk proses pelatihan selanjutnya. Fungsi iter_images() menelusuri struktur folder di dalam direktori data/train, di mana setiap subfolder dianggap sebagai satu kelas atau identitas, dan menghasilkan pasangan berupa path file gambar serta label kelasnya. Fungsi build_matrix() kemudian memanggil embed_from_path() untuk setiap gambar guna menghasilkan embedding FaceNet; embedding yang valid disimpan ke dalam list X, sedangkan gambar yang tidak terdeteksi wajahnya dicatat dalam list bad. Pada akhirnya, embedding X dan label y dikonversi menjadi array NumPy dan disimpan sebagai X_train.npy dan y_train.npy, sambil menampilkan jumlah embedding yang berhasil dan jumlah gambar yang gagal diproses. Script ini merupakan langkah penting untuk menyiapkan data bagi model klasifikasi wajah berbasis FaceNet.

4. train_classifier.py
   Kode Program :
  ```python
  #train_classifier.py
  import numpy as np
  from sklearn.svm import SVC
  from sklearn.model_selection import cross_val_score
  from sklearn.preprocessing import StandardScaler
  from sklearn.pipeline import Pipeline
  import joblib
  
  X = np.load("X_train.npy")
  y = np.load("y_train.npy", allow_pickle=True)
  
  #Pipeline: standardize -> SVM (RBF)
  clf = Pipeline([("scaler", StandardScaler()),
      ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                  class_weight="balanced"))
  ])
  
  scores = cross_val_score(clf, X, y, cv=2, scoring="accuracy")
  print("CV acc mean:", scores.mean(), "±", scores.std())
  
  clf.fit(X, y)
  joblib.dump(clf, "facenet_svm.joblib")
  
  print("Model disimpan ke facenet_svm.joblib")
  ```

  Analisis:
  Kode train_classifier.py digunakan untuk melatih model klasifikasi wajah berbasis SVM menggunakan embedding yang telah dihasilkan sebelumnya. Pertama, script memuat data embedding (X_train.npy) dan label identitas (y_train.npy). Model kemudian dibangun menggunakan pipeline yang berisi dua tahap: normalisasi fitur dengan StandardScaler dan klasifikasi menggunakan SVM dengan kernel RBF, lengkap dengan parameter C, gamma, serta pengaturan class_weight='balanced' untuk menangani jumlah data per kelas yang tidak seimbang. Sebelum pelatihan akhir, performa model diuji menggunakan cross_val_score dengan 2-fold cross-validation untuk memperoleh estimasi akurasi. Setelah itu, model dilatih menggunakan seluruh data dan disimpan ke file facenet_svm.joblib menggunakan joblib.dump(). Script ini merupakan tahap penting untuk membangun model pengenalan wajah yang siap digunakan untuk prediksi.

5. predict_one.py
   Kode Program :
  ```python
  #predict_one.py
  import joblib
  from utils_facenet import embed_from_path
  import numpy as np
  
  clf = joblib.load("facenet_svm.joblib")
  
  def predict_image(path, unknown_threshold=0.55):
      emb = embed_from_path(path)
      if emb is None:
          return "NO_FACE", 0.0
  
      proba = clf.predict_proba([emb])[0]       # probabilitas per kelas
      idx = int(np.argmax(proba))
      label = clf.classes_[idx]
      conf  = float(proba[idx])
  
      #optional rejection: tandai "unknown" bila confidence rendah
      if conf < unknown_threshold:
          return "UNKNOWN", conf
  
      return label, conf
  
  if __name__ == "__main__":
      test_img = "data/val/Isna/i4.webp"  # ganti sesuai data Anda
      label, conf = predict_image(test_img)
      print(f"Prediksi: {label} (conf={conf:.3f})")
  ```

  Analisis :
  Kode predict_one.py digunakan untuk melakukan prediksi identitas wajah pada satu gambar menggunakan model SVM yang telah dilatih sebelumnya. Program memuat model facenet_svm.joblib, lalu fungsi predict_image() mengambil embedding wajah dari gambar menggunakan embed_from_path(). Jika wajah tidak terdeteksi, fungsi langsung mengembalikan label "NO_FACE". Jika embedding berhasil dibuat, model menghitung probabilitas untuk setiap kelas, memilih kelas dengan nilai terbesar sebagai prediksi, dan mengambil confidence-nya. Script ini juga menyediakan mekanisme penolakan otomatis: jika confidence berada di bawah threshold tertentu (misalnya 0.55), gambar diberi label "UNKNOWN" untuk mencegah prediksi salah. Pada bagian utama program, satu gambar diuji dan hasil prediksi beserta tingkat kepercayaannya ditampilkan. 

6. eval_folder.py
   Kode Program :
  ```python
  #eval_folder.py
  import os, glob, numpy as np, joblib
  from collections import defaultdict
  from utils_facenet import embed_from_path
  
  clf = joblib.load("facenet_svm.joblib")
  
  def predict_emb(emb):
      proba = clf.predict_proba([emb])[0]
      idx = int(np.argmax(proba))
      return clf.classes_[idx], float(proba[idx])
  
  root = "data/val"
  
  y_true, y_pred = [], []
  per_cls = defaultdict(lambda: {"ok": 0, "total": 0})
  
  for cls in sorted(os.listdir(root)):
      pdir = os.path.join(root, cls)
      if not os.path.isdir(pdir):
          continue
  
      for p in glob.glob(os.path.join(pdir, "*")):
          emb = embed_from_path(p)
          if emb is None:
              continue
  
          pred, conf = predict_emb(emb)
          y_true.append(cls)
          y_pred.append(pred)
  
          per_cls[cls]["total"] += 1
          per_cls[cls]["ok"]    += int(pred == cls)
  
  acc = np.mean([t == p for t, p in zip(y_true, y_pred)])
  print("Accuracy:", acc)
  
  for c, st in per_cls.items():
      if st["total"] > 0:
          print(f"{c}: {st['ok']}/{st['total']} = {st['ok']/st['total']:.3f}")
  ```

  Analisis :
  Kode eval_folder.py digunakan untuk mengevaluasi akurasi model FaceNet + SVM pada seluruh gambar di folder validasi. Script ini memuat model facenet_svm.joblib, lalu membaca setiap folder dalam direktori data/val, di mana setiap folder dianggap sebagai label kelas sebenarnya. Untuk setiap gambar, embedding wajah dihasilkan menggunakan embed_from_path(), kemudian diprediksi dengan fungsi predict_emb(). Label asli (y_true) dan label prediksi (y_pred) disimpan untuk perhitungan akurasi keseluruhan. Selain itu, script juga menghitung akurasi per kelas menggunakan struktur per_cls, yang mencatat jumlah prediksi benar dan total gambar per identitas. Setelah semua gambar diproses, program mencetak akurasi global dan akurasi per kelas, sehingga memudahkan untuk melihat performa model untuk setiap orang dalam dataset. Dengan demikian, script ini berfungsi sebagai alat evaluasi untuk mengetahui seberapa baik model mengenali wajah pada data pengujian.

7. train_knn.py
   Kode Program :
  ```python
  #train_knn.py
  import numpy as np, joblib
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.preprocessing import StandardScaler
  from sklearn.pipeline import Pipeline
  
  X = np.load("X_train.npy")
  y = np.load("y_train.npy", allow_pickle=True)
  
  clf = Pipeline([ ("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=3, metric="euclidean")) ])
  
  clf.fit(X, y)
  joblib.dump(clf, "facenet_knn.joblib")
  print("Saved facenet_knn.joblib")
  ```

  Analisis :
  Kode train_knn.py digunakan untuk melatih model pengenalan wajah berbasis algoritma KNN menggunakan embedding FaceNet yang telah dibuat sebelumnya. Script ini memuat data embedding (X_train.npy) dan label identitas (y_train.npy), kemudian membangun pipeline yang terdiri dari StandardScaler untuk menormalkan fitur dan KNeighborsClassifier dengan 3 tetangga terdekat menggunakan jarak Euclidean. Setelah model dilatih dengan data tersebut, model disimpan ke file facenet_knn.joblib menggunakan joblib.dump(). Model KNN ini dapat menjadi alternatif SVM karena lebih sederhana dan cepat untuk pelatihan, serta cocok jika jumlah data per kelas tidak terlalu besar.

8. verify_cli.py
   Kode Program :
  ```python
  #verify_cli.py
  import argparse
  from utils_facenet import embed_from_path, cosine_similarity
  
  if __name__ == "__main__":
      ap = argparse.ArgumentParser()
      ap.add_argument("img1")
      ap.add_argument("img2")
      ap.add_argument("--th", type=float, default=0.85)
      args = ap.parse_args()
      
      e1 = embed_from_path(args.img1)
      e2 = embed_from_path(args.img2)
  
      if e1 is None or e2 is None:
          print("Wajah tidak terdeteksi pada salah satu gambar.")
      else:
          sim = cosine_similarity(e1, e2)
          print(
              f"Similarity={sim:.4f}  ->  {'MATCH' if sim>=args.th else 'NO MATCH'} "
              f"(th={args.th})"
          )
  ```
    
  Analisis :
  Kode verify_cli.py merupakan tool verifikasi wajah berbasis command-line yang memungkinkan pengguna membandingkan dua gambar melalui terminal. Script ini menggunakan argparse untuk menerima dua path gambar dan opsi threshold kemiripan dari pengguna. Setelah dijalankan, program mengekstraksi embedding wajah dari kedua gambar menggunakan embed_from_path(), lalu menghitung tingkat kemiripan menggunakan cosine_similarity(). Jika salah satu gambar tidak terdeteksi wajahnya, program menampilkan pesan error; jika berhasil, nilai similarity dicetak bersama hasil apakah kedua wajah dianggap “MATCH” atau “NO MATCH” berdasarkan threshold yang diberikan. Dengan demikian, kode ini menyediakan cara cepat dan praktis untuk menguji kemiripan dua wajah langsung dari CLI tanpa perlu membuka interface tambahan.

