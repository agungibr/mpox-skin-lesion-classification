# Monkeypox Skin Lesion Detection Using Deep Learning

![cover](https://github.com/user-attachments/assets/e329ab76-603f-4c6a-aec2-d0980b230699)

## Introduction
Penularan kasus MonkeyPox mengalami kenaikan yang sangat signifikan di beberapa negara Afrika, Eropa, Amerika, hingga Asia. Penularan kasus MonkeyPox juga dikhawatirkan akan menjadi kasus pandemik yang meluas pada semua negara seperti kasus Covid19 pada tahun 2020. Untuk mencegah penularan, penular (carrier) harus dapat diidentifikasi sejak awal sebelum menularkan kepada orang lain di dekatnya. Dengan memanfaatkan penginderaan jarak jauh seperti kamera berkualitas tinggi untuk melihat adanya ciri-ciri carrier pada masyarakat dapat memberikan peringatan dini untuk orang disekitarnya. Tetapi, diperlukan ahli yang dapat menentukan apakah ciri-ciri pada kulit dapat dikategorikan sebagai MonkeyPox ataupun tidak. Dengan keterbatasan ahli untuk terus menerus melihat dan menilai ciri-ciri kulit menjadikan penentuan carrier lebih sulit dilakukan. Diperlukan sebuah sistem yang dapat mengklasifikasikan ciri-ciri kulit yang sehat, MonkeyPox ataupun penyakit lain dengan **efisien, presisi, dan akurasi yang baik.** Diberikan dataset yang dapat di akses pada link berikut: [Source](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20/data)

## Data
Dataset terbagi menjadi dua kategori, yaitu:
1. Original Images

Original images terdiri dari 5 folds (lipatan) yang di dalamnya terdapat folder Train, Valid, dan Test. Masing-masing berisikan citra dari kulit dengan 6 kelas berbeda.

2. Augmented Images

Augmented images terdiri dari 5 folds (lipatan) yang di dalamnya hanya terdapat folder Train untuk masing-masing kelas.

## Commands
Untuk melatih model, dapat digunakan perintah berikut:
```python
python3 train.py
```
atau
```python
python train.py
```
Untuk melakukan evaluasi, dapat digunakan perintah berikut:
```python
python3 test.py
```
atau
```python
python test.py
```

## Model
![architecture](https://github.com/user-attachments/assets/7c2e6d50-90d1-4912-90d0-81abb1c3f296)

Disini digunakan Convolutional Neural Network (CNN) model pada kasus ini yang dipakai untuk memproses gambar dan mengklasifikasikan ke dalam satu dari enam kategori. Input dari gambar ini yaitu berdimensi `32 x 32` pixel dan 3 warna (RGB). Model ini mempunyai dua *block* yang berisi *Convolution, ReLU activation,* dan *MaxPooling*. Pada block pertama yang berisi *Convolution* yang berisi 20 filters yang masing-masing berukuran `5 x 5`. *MaxPooling* mengurangi spatial dimensions dengan mengambil maximum value pada setiap `2 x 2`. Lalu pada *block* kedua *Convolution* apply 50 filters dengan ukuran `5 x 5` ke output layer sebelumnya. Lalu *MaxPooling* mengambil nilai maximum dari setiap area `2 x 2`. Setelah itu dilakukan *flattened* menjadi vektor 1D untuk mempersiapkan fully connected (Dense) layer. Fully connected (fc) layer mengambil vektor 1D dan mengubahnya menjadi vektor dengan ukuran 6, sesuai dengan kelas dari data.

## Result
1. Cross Entropy Loss Function
![training](https://github.com/user-attachments/assets/01335a33-24c1-4130-81db-654c3cec8c71)

Loss function adalah sebuah fungsi yang digunakan untuk menghitung perbedaan antara output yang dibuat oleh algoritma saat ini dengan output yang diharapkan. Jika prediksi model mengandung banyak kesalahan, maka loss function akan menghasilkan angka yang lebih tinggi dan sebaliknya. Salahsatu yang digunakan dalam kasus ini yaitu Cross Entropy. Cross Entropy mengukur perbedaan antara dua distribusi probabilitas untuk variable acak tertentu. Dari hasil gambar diatas dengan `15 Epoch` dan garis merah menunjukan loss function training dan garis biru menunjukan loss function validation, dapat dilihat bahwa nilai loss cukup tinggi yang menunjukkan bahwa model memiliki tingkat kesalahan prediksi yang besar. Namun seiring dengan bertambahnya epoch nilai loss menurun secara konsisten menunjukkan bahwa model berhasil belajar pola dari data dan mengurangi kesalahan prediksi. Hasil ini berlaku untuk training dan validation yang menunjukan model memiliki performa yang baik karena tidak terjadi overfitting.

2. Accuracy, Precision, Recall, dan F1 Score
![f1score](https://github.com/user-attachments/assets/9222f182-98d3-452b-8f2f-fa2d8eb1243c)

Accuracy score mengukur proporsi prediksi yang benar dari seluruh data. Didapatkan nilai 0.99725 yang berarti model benar dalam 99.725% dari semua prediksi. Precision mengukur proporsi prediksi positif yang benar. Fokusnya adalah seberapa andal model saat memprediksi kelas positif. Didapatkan nilai 0.99727 yang berarti dari semua prediksi positif model 99.727% benar. Kemudian ada Recall atau sensitivity yang mengukur seberapa baik model menangkap semua sampel positif sebenarnya. Fokusnya pada False Negatives. Didapatkan nilai 0.99725 yang berarti model berhasil menangkap 99.725% dari seluruh data positif yang sebenarnya. Terakhir ada F1 Score yang menghitung rata-rata harmonik dari precision dan recall. Didapatkan nilai 0.99724 yang berarti menunjukkan keseimbangan yang baik antara precision dan recall.

3. Confusion Matrix
![confusion_matrix](https://github.com/user-attachments/assets/86122326-340b-4cbb-af92-de8e3f28c7cf)

Confusion matrix adalah representasi tabular dari kinerja model klasifikasi. Matrix ini menunjukkan perbandingan antara nilai aktual (label sebenarnya) dengan prediksi yang dihasilkan oleh model. Dalam confusion matrix, setiap baris merepresentasikan class sebenarnya (true labels), sedangkan setiap kolom merepresentasikan class prediksi (predicted labels). Diagonal (kiri atas ke kanan bawah) menunjukkan jumlah prediksi yang benar untuk setiap class, sedangkan elemen lainnya menunjukkan jumlah prediksi yang salah.

Dari gambar diatas menunjukan semua class memiliki nilai prediksi sempurna 1.0 kecuali pada class HFMD yang memiliki nilai 0.98 yang menunjukkan bahwa ada beberapa kesalahan pada prediksi untuk class ini. Ada nilai 0.02 pada baris HFMD di kolom class lain, yang berarti 2% dari data class HFMD salah diklasifikasikan ke class lain.

4. ROC-AUC
![roc_auc](https://github.com/user-attachments/assets/77478c74-84d9-494b-86b8-f31a9fee4a84)

ROC (Receiver Operating Characteristic) adalah grafik yang menunjukkan performa model klasifikasi pada berbagai nilai ambang (threshold). Grafik ini memplot True Positive Rate (TPR) di sumbu y dan False Positive Rate (FPR) di sumbu x. AUC (Area Under the Curve) adalah ukuran luas di bawah kurva ROC. Dari gambar diatas dapat dilihat semua kelas hampir mendekati 1 (100%), kecuali Monkeypox (AUC ≈ 0.99). Hal ini menunjukkan model memiliki performa yang sangat baik dalam membedakan setiap kelas. Garis putus-putus (AUC = 0.50) menunjukkan performa tebak acak