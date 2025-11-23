# 🤖 Machine Learning Codes and Datasets

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![R](https://img.shields.io/badge/R-4.0+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)
![Stars](https://img.shields.io/github/stars/CemRoot/Machine-Learning-Codes-and-Datasets?style=social)

**A comprehensive collection of machine learning algorithms, datasets, and implementations**

[English](#english) | [Türkçe](#turkce)

</div>

---

<a name="english"></a>

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithms Covered](#algorithms-covered)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

Welcome to the **Machine Learning Codes and Datasets** repository! This comprehensive resource is designed for machine learning practitioners, students, and enthusiasts who want to learn, practice, and master various ML algorithms and techniques.

### 🎓 What You Will Learn

- ✅ Implementation of **40+ machine learning algorithms** from scratch
- ✅ Data preprocessing and feature engineering techniques
- ✅ Model evaluation and performance optimization
- ✅ Deep learning with neural networks
- ✅ Natural language processing fundamentals
- ✅ Dimensionality reduction methods
- ✅ Best practices in machine learning workflows

## ⭐ Features

- 📚 **Comprehensive Coverage**: 11 major ML domains with 40+ algorithms
- 💻 **Dual Language Support**: Python and R implementations
- 📊 **Real-world Datasets**: Curated datasets for each algorithm
- 📝 **Interactive Notebooks**: Jupyter notebooks with detailed explanations
- 🔬 **Production-ready Code**: Clean, documented, and modular code
- 🎨 **Visualization**: Beautiful plots and charts for better understanding
- 🌍 **Bilingual Documentation**: Full English and Turkish support

## 📁 Repository Structure

```
Machine-Learning-Codes-and-Datasets/
│
├── Part 1 - Data Preprocessing/
│   └── Data cleaning, transformation, and feature scaling
│
├── Part 2 - Regression/
│   ├── Simple Linear Regression
│   ├── Multiple Linear Regression
│   ├── Polynomial Regression
│   ├── Support Vector Regression (SVR)
│   ├── Decision Tree Regression
│   └── Random Forest Regression
│
├── Part 3 - Classification/
│   ├── Logistic Regression
│   ├── K-Nearest Neighbors (K-NN)
│   ├── Support Vector Machine (SVM)
│   ├── Kernel SVM
│   ├── Naive Bayes
│   ├── Decision Tree Classification
│   └── Random Forest Classification
│
├── Part 4 - Clustering/
│   ├── K-Means Clustering
│   └── Hierarchical Clustering
│
├── Part 5 - Association Rule Learning/
│   ├── Apriori Algorithm
│   └── Eclat Algorithm
│
├── Part 6 - Reinforcement Learning/
│   ├── Upper Confidence Bound (UCB)
│   └── Thompson Sampling
│
├── Part 7 - Natural Language Processing/
│   └── Text preprocessing and sentiment analysis
│
├── Part 8 - Deep Learning/
│   ├── Artificial Neural Networks (ANN)
│   └── Convolutional Neural Networks (CNN)
│
├── Part 9 - Dimensionality Reduction/
│   ├── Principal Component Analysis (PCA)
│   ├── Linear Discriminant Analysis (LDA)
│   └── Kernel PCA
│
├── Part 10 - Model Selection & Boosting/
│   ├── k-Fold Cross Validation
│   ├── Grid Search
│   └── XGBoost
│
└── ML_cheatsheet.pdf
```

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:

- **Python**: 3.7 or higher
- **R**: 4.0 or higher (optional, for R implementations)
- **pip**: Python package manager
- **Git**: Version control system

### Required Knowledge

- Basic understanding of Python programming
- Fundamentals of linear algebra and statistics
- Familiarity with NumPy and Pandas (recommended)

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/CemRoot/Machine-Learning-Codes-and-Datasets.git
cd Machine-Learning-Codes-and-Datasets
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# For Python venv
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate

# OR using Conda
conda create -n ml_env python=3.9
conda activate ml_env
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Launch Jupyter Notebook

```bash
jupyter notebook
```

## 💡 Usage

### Quick Start Example

Here's a simple example of using linear regression from this repository:

```python
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Visualize results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

### Running Individual Algorithms

Navigate to the specific algorithm directory and run the Python script or Jupyter notebook:

```bash
# Example: Running K-Means Clustering
cd "Part 4 - Clustering/Section 24 - K-Means Clustering/Python"
python k_means_clustering.py

# Or open the Jupyter notebook
jupyter notebook k_means_clustering.ipynb
```

## 🧠 Algorithms Covered

<details>
<summary><b>Part 1: Data Preprocessing</b></summary>

- Handling Missing Data
- Encoding Categorical Data
- Feature Scaling (Standardization & Normalization)
- Train/Test Split

</details>

<details>
<summary><b>Part 2: Regression (6 Algorithms)</b></summary>

- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Support Vector Regression (SVR)
- Decision Tree Regression
- Random Forest Regression

</details>

<details>
<summary><b>Part 3: Classification (7 Algorithms)</b></summary>

- Logistic Regression
- K-Nearest Neighbors (K-NN)
- Support Vector Machine (SVM)
- Kernel SVM
- Naive Bayes
- Decision Tree Classification
- Random Forest Classification

</details>

<details>
<summary><b>Part 4: Clustering (2 Algorithms)</b></summary>

- K-Means Clustering
- Hierarchical Clustering

</details>

<details>
<summary><b>Part 5: Association Rule Learning (2 Algorithms)</b></summary>

- Apriori
- Eclat

</details>

<details>
<summary><b>Part 6: Reinforcement Learning (2 Algorithms)</b></summary>

- Upper Confidence Bound (UCB)
- Thompson Sampling

</details>

<details>
<summary><b>Part 7: Natural Language Processing</b></summary>

- Bag of Words Model
- Text Preprocessing
- Sentiment Analysis

</details>

<details>
<summary><b>Part 8: Deep Learning (2 Types)</b></summary>

- Artificial Neural Networks (ANN)
- Convolutional Neural Networks (CNN)

</details>

<details>
<summary><b>Part 9: Dimensionality Reduction (3 Algorithms)</b></summary>

- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)
- Kernel PCA

</details>

<details>
<summary><b>Part 10: Model Selection & Boosting</b></summary>

- k-Fold Cross Validation
- Grid Search for Hyperparameter Tuning
- XGBoost

</details>

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Primary programming language |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical computing |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Data manipulation |
| ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Machine learning algorithms |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) | Deep learning framework |
| ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) | Neural networks API |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) | Data visualization |
| ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) | Interactive notebooks |
| ![R](https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white) | Statistical computing |

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

1. **Fork** the repository
2. **Create** a new branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

### Ways to Contribute

- 🐛 Report bugs and issues
- 💡 Suggest new features or algorithms
- 📝 Improve documentation
- 🧪 Add new datasets
- ✨ Enhance existing implementations
- 🌍 Translate documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by various machine learning courses and resources
- Thanks to all contributors who have helped improve this repository
- Special thanks to the open-source ML community

## 📞 Contact & Support

- **Repository**: [GitHub](https://github.com/CemRoot/Machine-Learning-Codes-and-Datasets)
- **Issues**: [Report a Bug](https://github.com/CemRoot/Machine-Learning-Codes-and-Datasets/issues)
- **Discussions**: [Join the Discussion](https://github.com/CemRoot/Machine-Learning-Codes-and-Datasets/discussions)

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

If this repository helped you in your machine learning journey, please consider giving it a star ⭐

**Happy Learning! 🚀**

</div>

---

<a name="turkce"></a>

## 🇹🇷 Türkçe Dokümantasyon

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [Özellikler](#özellikler)
- [Depo Yapısı](#depo-yapısı)
- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Kapsanan Algoritmalar](#kapsanan-algoritmalar)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)

## 🎯 Genel Bakış

**Makine Öğrenmesi Kodları ve Veri Setleri** deposuna hoş geldiniz! Bu kapsamlı kaynak, makine öğrenmesi uygulayıcıları, öğrenciler ve meraklıları için çeşitli ML algoritmalarını ve tekniklerini öğrenmek, pratik yapmak ve ustalaşmak için tasarlanmıştır.

### 🎓 Neler Öğreneceksiniz

- ✅ **40+ makine öğrenmesi algoritmasının** sıfırdan implementasyonu
- ✅ Veri ön işleme ve özellik mühendisliği teknikleri
- ✅ Model değerlendirme ve performans optimizasyonu
- ✅ Sinir ağları ile derin öğrenme
- ✅ Doğal dil işleme temelleri
- ✅ Boyut azaltma yöntemleri
- ✅ Makine öğrenmesi iş akışlarında en iyi uygulamalar

## ⭐ Özellikler

- 📚 **Kapsamlı Kapsam**: 40+ algoritma ile 11 ana ML alanı
- 💻 **Çift Dil Desteği**: Python ve R implementasyonları
- 📊 **Gerçek Dünya Veri Setleri**: Her algoritma için özenle seçilmiş veri setleri
- 📝 **Etkileşimli Notebook'lar**: Detaylı açıklamalı Jupyter notebook'ları
- 🔬 **Üretime Hazır Kod**: Temiz, dokümante edilmiş ve modüler kod
- 🎨 **Görselleştirme**: Daha iyi anlama için güzel grafikler ve çizelgeler
- 🌍 **İki Dilli Dokümantasyon**: Tam İngilizce ve Türkçe destek

## 📁 Depo Yapısı

```
Machine-Learning-Codes-and-Datasets/
│
├── Part 1 - Veri Ön İşleme/
│   └── Veri temizleme, dönüştürme ve özellik ölçeklendirme
│
├── Part 2 - Regresyon/
│   ├── Basit Doğrusal Regresyon
│   ├── Çoklu Doğrusal Regresyon
│   ├── Polinom Regresyon
│   ├── Destek Vektör Regresyonu (SVR)
│   ├── Karar Ağacı Regresyonu
│   └── Rastgele Orman Regresyonu
│
├── Part 3 - Sınıflandırma/
│   ├── Lojistik Regresyon
│   ├── K-En Yakın Komşu (K-NN)
│   ├── Destek Vektör Makinesi (SVM)
│   ├── Kernel SVM
│   ├── Naive Bayes
│   ├── Karar Ağacı Sınıflandırması
│   └── Rastgele Orman Sınıflandırması
│
├── Part 4 - Kümeleme/
│   ├── K-Means Kümeleme
│   └── Hiyerarşik Kümeleme
│
├── Part 5 - Birliktelik Kuralı Öğrenimi/
│   ├── Apriori Algoritması
│   └── Eclat Algoritması
│
├── Part 6 - Pekiştirmeli Öğrenme/
│   ├── Üst Güven Sınırı (UCB)
│   └── Thompson Örneklemesi
│
├── Part 7 - Doğal Dil İşleme/
│   └── Metin ön işleme ve duygu analizi
│
├── Part 8 - Derin Öğrenme/
│   ├── Yapay Sinir Ağları (ANN)
│   └── Evrişimli Sinir Ağları (CNN)
│
├── Part 9 - Boyut Azaltma/
│   ├── Temel Bileşen Analizi (PCA)
│   ├── Doğrusal Diskriminant Analizi (LDA)
│   └── Kernel PCA
│
├── Part 10 - Model Seçimi ve Güçlendirme/
│   ├── k-Katlı Çapraz Doğrulama
│   ├── Izgara Araması
│   └── XGBoost
│
└── ML_cheatsheet.pdf
```

## 🔧 Gereksinimler

Başlamadan önce, aşağıdakilerin yüklü olduğundan emin olun:

- **Python**: 3.7 veya üzeri
- **R**: 4.0 veya üzeri (opsiyonel, R implementasyonları için)
- **pip**: Python paket yöneticisi
- **Git**: Versiyon kontrol sistemi

### Gerekli Bilgi

- Python programlama temel bilgisi
- Lineer cebir ve istatistik temelleri
- NumPy ve Pandas bilgisi (önerilir)

## 🚀 Kurulum

### Adım 1: Depoyu Klonlayın

```bash
git clone https://github.com/CemRoot/Machine-Learning-Codes-and-Datasets.git
cd Machine-Learning-Codes-and-Datasets
```

### Adım 2: Sanal Ortam Oluşturun (Önerilir)

```bash
# Python venv için
python -m venv ml_env
source ml_env/bin/activate  # Windows'ta: ml_env\Scripts\activate

# VEYA Conda kullanarak
conda create -n ml_env python=3.9
conda activate ml_env
```

### Adım 3: Bağımlılıkları Yükleyin

```bash
pip install -r requirements.txt
```

### Adım 4: Jupyter Notebook'u Başlatın

```bash
jupyter notebook
```

## 💡 Kullanım

### Hızlı Başlangıç Örneği

Bu depodan lineer regresyon kullanımına dair basit bir örnek:

```python
# Gerekli kütüphaneleri içe aktarın
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Veri setini yükleyin
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Veri setini bölün
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Modeli eğitin
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Tahmin yapın
y_pred = regressor.predict(X_test)

# Sonuçları görselleştirin
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title('Maaş vs Deneyim')
plt.xlabel('Deneyim Yılı')
plt.ylabel('Maaş')
plt.show()
```

### Tekil Algoritmaları Çalıştırma

Belirli bir algoritma dizinine gidin ve Python scriptini veya Jupyter notebook'unu çalıştırın:

```bash
# Örnek: K-Means Kümeleme çalıştırma
cd "Part 4 - Clustering/Section 24 - K-Means Clustering/Python"
python k_means_clustering.py

# Veya Jupyter notebook'u açın
jupyter notebook k_means_clustering.ipynb
```

## 🧠 Kapsanan Algoritmalar

<details>
<summary><b>Bölüm 1: Veri Ön İşleme</b></summary>

- Eksik Verilerin İşlenmesi
- Kategorik Verilerin Kodlanması
- Özellik Ölçeklendirme (Standardizasyon ve Normalizasyon)
- Eğitim/Test Ayrımı

</details>

<details>
<summary><b>Bölüm 2: Regresyon (6 Algoritma)</b></summary>

- Basit Doğrusal Regresyon
- Çoklu Doğrusal Regresyon
- Polinom Regresyon
- Destek Vektör Regresyonu (SVR)
- Karar Ağacı Regresyonu
- Rastgele Orman Regresyonu

</details>

<details>
<summary><b>Bölüm 3: Sınıflandırma (7 Algoritma)</b></summary>

- Lojistik Regresyon
- K-En Yakın Komşu (K-NN)
- Destek Vektör Makinesi (SVM)
- Kernel SVM
- Naive Bayes
- Karar Ağacı Sınıflandırması
- Rastgele Orman Sınıflandırması

</details>

<details>
<summary><b>Bölüm 4: Kümeleme (2 Algoritma)</b></summary>

- K-Means Kümeleme
- Hiyerarşik Kümeleme

</details>

<details>
<summary><b>Bölüm 5: Birliktelik Kuralı Öğrenimi (2 Algoritma)</b></summary>

- Apriori
- Eclat

</details>

<details>
<summary><b>Bölüm 6: Pekiştirmeli Öğrenme (2 Algoritma)</b></summary>

- Üst Güven Sınırı (UCB)
- Thompson Örneklemesi

</details>

<details>
<summary><b>Bölüm 7: Doğal Dil İşleme</b></summary>

- Kelime Çantası Modeli
- Metin Ön İşleme
- Duygu Analizi

</details>

<details>
<summary><b>Bölüm 8: Derin Öğrenme (2 Tip)</b></summary>

- Yapay Sinir Ağları (ANN)
- Evrişimli Sinir Ağları (CNN)

</details>

<details>
<summary><b>Bölüm 9: Boyut Azaltma (3 Algoritma)</b></summary>

- Temel Bileşen Analizi (PCA)
- Doğrusal Diskriminant Analizi (LDA)
- Kernel PCA

</details>

<details>
<summary><b>Bölüm 10: Model Seçimi ve Güçlendirme</b></summary>

- k-Katlı Çapraz Doğrulama
- Hiperparametre Ayarı için Izgara Araması
- XGBoost

</details>

## 🤝 Katkıda Bulunma

Topluluktan katkıları memnuniyetle karşılıyoruz! İşte nasıl yardımcı olabilirsiniz:

1. Depoyu **Fork** edin
2. Yeni bir branch **oluşturun** (`git checkout -b feature/HarikaBirOzellik`)
3. Değişikliklerinizi **commit** edin (`git commit -m 'Harika bir özellik ekle'`)
4. Branch'e **push** yapın (`git push origin feature/HarikaBirOzellik`)
5. Bir Pull Request **açın**

Davranış kurallarımız ve pull request gönderme süreci hakkında detaylar için lütfen [CONTRIBUTING.md](CONTRIBUTING.md) dosyasını okuyun.

### Katkı Yolları

- 🐛 Hata ve sorunları bildirin
- 💡 Yeni özellikler veya algoritmalar önerin
- 📝 Dokümantasyonu geliştirin
- 🧪 Yeni veri setleri ekleyin
- ✨ Mevcut implementasyonları iyileştirin
- 🌍 Dokümantasyonu çevirin

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🙏 Teşekkürler

- Çeşitli makine öğrenmesi kursları ve kaynaklarından esinlenilmiştir
- Bu depoyu geliştirmeye yardımcı olan tüm katkıda bulunanlara teşekkürler
- Açık kaynak ML topluluğuna özel teşekkürler

## 📞 İletişim ve Destek

- **Depo**: [GitHub](https://github.com/CemRoot/Machine-Learning-Codes-and-Datasets)
- **Sorunlar**: [Hata Bildirin](https://github.com/CemRoot/Machine-Learning-Codes-and-Datasets/issues)
- **Tartışmalar**: [Tartışmaya Katılın](https://github.com/CemRoot/Machine-Learning-Codes-and-Datasets/discussions)

---

<div align="center">

### ⭐ Faydalı bulduysanız bu depoyu yıldızlayın!

Bu depo makine öğrenmesi yolculuğunuzda size yardımcı olduysa, lütfen bir yıldız vermeyi düşünün ⭐

**Mutlu Öğrenmeler! 🚀**

</div>

---

## 📈 Project Stats

![GitHub last commit](https://img.shields.io/github/last-commit/CemRoot/Machine-Learning-Codes-and-Datasets)
![GitHub issues](https://img.shields.io/github/issues/CemRoot/Machine-Learning-Codes-and-Datasets)
![GitHub pull requests](https://img.shields.io/github/issues-pr/CemRoot/Machine-Learning-Codes-and-Datasets)
![GitHub forks](https://img.shields.io/github/forks/CemRoot/Machine-Learning-Codes-and-Datasets?style=social)

---

<div align="center">

**Made with ❤️ for the Machine Learning Community**

</div>
