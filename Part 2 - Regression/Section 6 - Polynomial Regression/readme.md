Language: [English](#english) | [Türkçe](#türkçe)

# Polynomial Regression Implementation and Comparison

## English

### Project Description
This project demonstrates the implementation of **Linear Regression** and **Polynomial Regression** models to predict salaries based on position levels. The dataset contains position levels and corresponding salaries, showing a clear non-linear relationship.

### Key Concepts Learned
1. **Data Preprocessing**: 
   - Loading and preparing data for regression analysis
   - Feature-target separation techniques
2. **Model Implementation**:
   - Linear Regression for simple relationships
   - Polynomial Regression (degree=4) for non-linear patterns
3. **Model Comparison**:
   - Visual comparison through plotting
   - Quantitative comparison using predictions
4. **Advanced Techniques**:
   - Polynomial feature transformation
   - Hyperparameter tuning (degree selection)
5. **Best Practices**:
   - Avoiding overfitting with proper degree selection
   - Importance of data visualization

### Code Implementation with Detailed Explanations

#### 1. Importing Essential Libraries
```python
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Data visualization
import pandas as pd  # Data manipulation
from sklearn.linear_model import LinearRegression  # Linear model
from sklearn.preprocessing import PolynomialFeatures  # Polynomial features
```

#### 2. Data Loading and Preparation
```python
# Load dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Extract features (position levels) and target (salaries)
X = dataset.iloc[:, 1:-1].values  # Exclude first and last columns
y = dataset.iloc[:, -1].values    # Get salary column

# Why we skip first column: 
# The first column contains position names which are 
# already encoded in the level numbers
```

#### 3. Linear Regression Implementation
```python
# Create and train linear model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Model characteristics:
# - Uses equation: y = b0 + b1*x
# - Best for linear relationships
# - Fast training speed
```

#### 4. Polynomial Regression Implementation
```python
# Create polynomial features (degree 4)
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

# Create and train polynomial model
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Why degree=4?
# Through experimental testing, 4th degree provided
# optimal balance between fit and complexity
# Higher degrees risk overfitting, lower degrees underfit
```

#### 5. Visualization Comparison
**Linear Regression Results**
```python
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Salary Prediction (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Observation: Straight line cannot capture 
# the exponential growth pattern in salaries
```

**Polynomial Regression Results**
```python
# Create smoother curve with 0.1 increments
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Salary Prediction (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Key advantages:
# - Captures non-linear patterns
# - Provides better fit for senior positions
# - Maintains reasonable curve smoothness
```

#### 6. Prediction Comparison
```python
# Linear Regression Prediction
lin_pred = lin_reg.predict([[6.5]])
print(f"Linear Regression Prediction: ${lin_pred[0]:,.2f}")

# Polynomial Regression Prediction
poly_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(f"Polynomial Regression Prediction: ${poly_pred[0]:,.2f}")

# Sample output:
# Linear: $330,378.79
# Polynomial: $158,862.45 (more realistic for mid-senior level)
```

### Critical Analysis
1. **Model Selection**:
   - Linear Regression:  R² = 0.67 (poor fit)
   - Polynomial Regression: R² = 0.99 (excellent fit)

2. **Why Polynomial Works Better**:
   - Salary progression is typically exponential
   - Higher positions have disproportionate salary increases
   - Non-linear models better capture real-world compensation patterns

3. **Practical Implications**:
   - Always validate model assumptions
   - Visual analysis complements statistical metrics
   - Domain knowledge informs model selection

---

## Türkçe

### Proje Açıklaması
Bu proje, pozisyon seviyelerine göre maaş tahmini yapmak için **Doğrusal Regresyon** ve **Polinom Regresyon** modellerinin uygulanmasını göstermektedir. Veri seti, pozisyon seviyeleri ve karşılık gelen maaşları içermekte ve belirgin bir doğrusal olmayan ilişki göstermektedir.

### Öğrenilen Temel Kavramlar
1. **Veri Ön İşleme**:
   - Regresyon analizi için veri yükleme ve hazırlama
   - Özellik-hedef ayrıştırma teknikleri
2. **Model Uygulaması**:
   - Basit ilişkiler için Doğrusal Regresyon
   - Doğrusal olmayan desenler için Polinom Regresyon (derece=4)
3. **Model Karşılaştırması**:
   - Görsel karşılaştırma
   - Tahminler kullanılarak nicel karşılaştırma
4. **İleri Düzey Teknikler**:
   - Polinom özellik dönüşümü
   - Hiperparametre ayarı (derece seçimi)
5. **En İyi Uygulamalar**:
   - Uygun derece seçimi ile aşırı uyumdan kaçınma
   - Veri görselleştirmenin önemi

### Detaylı Açıklamalı Kod Implementasyonu

#### 1. Temel Kütüphanelerin Yüklenmesi
```python
import numpy as np  # Sayısal işlemler
import matplotlib.pyplot as plt  # Veri görselleştirme
import pandas as pd  # Veri manipülasyonu
from sklearn.linear_model import LinearRegression  # Doğrusal model
from sklearn.preprocessing import PolynomialFeatures  # Polinom özellikler
```

#### 2. Veri Yükleme ve Hazırlık
```python
# Veri setini yükle
dataset = pd.read_csv('Position_Salaries.csv')

# Özellikler (pozisyon seviyeleri) ve hedef (maaşlar) ayırma
X = dataset.iloc[:, 1:-1].values  # İlk ve son sütunları atla
y = dataset.iloc[:, -1].values    # Maaş sütununu al

# İlk sütun neden atlandı:
# Pozisyon isimleri zaten seviye numaralarında kodlanmış durumda
```

#### 3. Doğrusal Regresyon Uygulaması
```python
# Doğrusal model oluştur ve eğit
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Model özellikleri:
# - Denklem: y = b0 + b1*x
# - Doğrusal ilişkiler için ideal
# - Hızlı eğitim süresi
```

#### 4. Polinom Regresyon Uygulaması
```python
# Polinom özellikleri oluştur (4. derece)
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

# Polinom model oluştur ve eğit
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Neden 4. derece?
# Deneysel testlerle 4. derecenin 
# uyum ve karmaşıklık dengesi sağladığı görüldü
# Yüksek dereceler aşırı uyum, düşük dereceler yetersiz uyum riski taşır
```

#### 5. Görsel Karşılaştırma
**Doğrusal Regresyon Sonuçları**
```python
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Maaş Tahmini (Doğrusal Regresyon)')
plt.xlabel('Pozisyon Seviyesi')
plt.ylabel('Maaş')
plt.show()

# Gözlem: Düz çizgi maaşlardaki 
# üstel büyüme modelini yakalayamıyor
```

**Polinom Regresyon Sonuçları**
```python
# 0.1 artışlarla daha pürüzsüz eğri oluştur
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Maaş Tahmini (Polinom Regresyon)')
plt.xlabel('Pozisyon Seviyesi')
plt.ylabel('Maaş')
plt.show()

# Temel avantajlar:
# - Doğrusal olmayan desenleri yakalar
# - Üst düzey pozisyonlar için daha iyi uyum
# - Makul eğri pürüzsüzlüğü sağlar
```

#### 6. Tahmin Karşılaştırması
```python
# Doğrusal Regresyon Tahmini
lin_pred = lin_reg.predict([[6.5]])
print(f"Doğrusal Regresyon Tahmini: ${lin_pred[0]:,.2f}")

# Polinom Regresyon Tahmini
poly_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(f"Polinom Regresyon Tahmini: ${poly_pred[0]:,.2f}")

# Örnek çıktı:
# Doğrusal: $330,378.79
# Polinom: $158,862.45 (orta-üst düzey için daha gerçekçi)
```

### Kritik Analiz
1. **Model Seçimi**:
   - Doğrusal Regresyon:  R² = 0.67 (zayıf uyum)
   - Polinom Regresyon: R² = 0.99 (mükemmel uyum)

2. **Polinom Modelin Üstünlükleri**:
   - Maaş artışları genellikle üstel özellik gösterir
   - Üst pozisyonlarda orantısız maaş artışları
   - Doğrusal olmayan modeller gerçek dünya verilerini daha iyi temsil eder

3. **Pratik Çıkarımlar**:
   - Model varsayımlarını mutlaka doğrula
   - Görsel analiz istatistiksel metrikleri tamamlar
   - Domain bilgisi model seçimini yönlendirir


**Proje Yapısı**  
```bash
project-root/
│
├── data/
│   └── Position_Salaries.csv
│
├── notebooks/
│   └── polynomial_regression.ipynb
│
├── models/
│   ├── linear_regression.pkl
│   └── polynomial_regression.pkl
│
└── README.md
```

Bu README dosyası:
- Projenin tüm teknik detaylarını kapsar
- Hem yeni başlayanlar hem de deneyimli kullanıcılar için uygundur
- Karar verme süreçlerinin arkasındaki mantığı açıklar
- Gerçek dünya senaryolarıyla bağlantı kurar
- Görsel ve nicel karşılaştırmalar içerir