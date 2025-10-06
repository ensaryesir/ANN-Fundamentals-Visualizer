# Decision Boundary Classification Visualizer (Streamlit)

Bu proje, temel makine öğrenmesi kütüphaneleri (scikit-learn, TensorFlow, PyTorch vb.) kullanılmadan, sıfırdan yazılmış basit sınıflandırma algoritmalarının (Perceptron ve Delta Kuralı) karar sınırlarını interaktif olarak görselleştirmeyi amaçlar. Uygulama Streamlit ile geliştirilmiştir ve NumPy + Matplotlib kullanır.

## Özellikler
- Perceptron ve DeltaRule (Bipolar Sigmoid) sınıflandırıcıları
- Streamlit arayüzü ile nokta ekleme ve veri yönetimi
- Eğitim/test ayrımı (rastgele), epoch bazlı animasyonlu karar sınırı çizimi
- Test doğruluğunun eğitim sonrasında gösterimi

## Kurulum

1) Gerekli bağımlılıkları yükleyin:
```bash
pip install -r decision-boundary-classification/requirements.txt
```

2) Uygulamayı başlatın:
```bash
streamlit run decision-boundary-classification/app.py
```

Alternatif: Klasör içinde çalıştırmak isterseniz
```bash
cd decision-boundary-classification
pip install -r requirements.txt
streamlit run app.py
```

## Kullanım
1) Kenar çubuğundan algoritmayı (Perceptron/DeltaRule), test oranını, öğrenme oranını ve epoch sayısını seçin.
2) Nokta eklemek için ana ekranda x1 ve x2 değerlerini girin, kenar çubuğundan noktanın sınıfını seçin ve "Add Point" tuşuna basın.
3) Noktaları temizlemek için kenar çubuğundaki "Clear Points" tuşunu kullanın.
4) "Train Model" tuşuna basarak eğitimi başlatın. Eğitim sırasında her epoch sonunda karar sınırı güncellenerek grafikte animasyonlu olarak gösterilir.
5) Eğitim tamamlanınca test doğruluğu ekranda görünür.

## Proje Yapısı
```
decision-boundary-classification/
├── app.py              # Ana uygulama (orchestrator)
├── algorithms.py       # BaseClassifier, Perceptron, DeltaRule
├── data_manager.py     # Streamlit session_state ile veri yönetimi
├── ui_components.py    # Kenar çubuğu ve UI bileşenleri
├── visualizer.py       # Matplotlib çizimleri ve karar sınırı
└── requirements.txt    # Bağımlılıklar
```

## Teknik Notlar
- Etiketler: {-1, +1}
- Bias, artırılmış vektör (X'e 1 ekleyerek) ile temsil edilir.
- `Perceptron.fit(X, y)` ve `DeltaRule.fit(X, y)` generator olarak tasarlanmıştır ve her epoch sonunda ağırlıkları `yield` eder.
- DeltaRule için aktivasyon: Bipolar Sigmoid `f(net) = 2/(1+exp(-net)) - 1` ve türevi `f'(net) = 0.5 * (1 - output^2)`.

## Sık Karşılaşılan Sorunlar
- `pip install requirements.txt` hatası: `-r` bayrağı ile kullanın: `pip install -r requirements.txt`.
- Streamlit API değişikliği: `st.experimental_rerun()` yerine `st.rerun()` kullanılmaktadır.
- Modül bulunamadı hataları: Uygulamayı `decision-boundary-classification` klasöründen çalıştırın veya proje kökünden tam yol ile çalıştırın.

## Lisans
Bu proje eğitim amaçlıdır.