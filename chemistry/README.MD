# Chemistry Module Architecture

Bu klasör, çimento prosesindeki kimyasal dönüşümleri **faz tabanlı (phase-based)** bir mimari ile modeller.

Eski yaklaşımda reaksiyonlar;
- dönüşüm katsayıları (`X_*`)
- toplam ürün kütle akışları (`m_dot_*`)
- bağımsız kimyasal değişkenler

üzerinden takip edilirken, yeni mimaride tüm kimyasal durum doğrudan:


üzerinden yönetilir.

---

# Genel Klasör Yapısı

# Genel Klasör Yapısı


chemistry/

├── base.py
│ └── Tüm reaksiyon modellerinin ortak altyapısı
│ - Arrhenius kinetiği
│ - sıcaklık aktivasyon aralığı
│ - reaksiyona giren faz miktarı hesabı
│ - reaksiyon ısı hesabı
│
├── phases.py
│ └── Kimyasal faz veri modelleri
│
│ SolidPhases:
│ H2O
│ CaCO3
│ CaO
│ SiO2
│ Al2O3
│ Fe2O3
│ C2S
│ C3S
│ C3A
│ C4AF
│
│ GasPhases:
│ CO2
│ H2O
│
├── composition.py
│ └── Başlangıç hammadde kompozisyonu
│ Raw meal içerisindeki faz dağılımını tanımlar.
│
├── drying.py
│ └── Serbest suyun uzaklaştırılması
│
├── dehydroxylation.py
│ └── Bağlı suyun ayrışması
│
├── calcination.py
│ └── Kalsinasyon reaksiyonu
│
│ CaCO3 → CaO + CO2
│
├── belite.py
│ └── Belit oluşumu
│
│ 2CaO + SiO2 → C2S
│
├── alite.py
│ └── Alit oluşumu
│
│ C2S + CaO → C3S
│
├── c3a.py
│ └── Kalsiyum aluminat oluşumu
│
│ 3CaO + Al2O3 → C3A
│
├── c4af.py
│ └── Ferrit faz oluşumu
│
│ CaO + Al2O3 + Fe2O3 → C4AF
│
└── reactions.py
└── Tüm reaksiyonların orkestrasyonu

    ChemistryModel:
        - drying
        - dehydroxylation
        - calcination
        - belite
        - alite
        - C3A
        - C4AF


    
---

# Faz Tabanlı Veri Akışı

Kimyasal durum tek tek değişkenler halinde değil, fazlar üzerinden taşınır.

RAW_MEAL_COMPOSITION

    ↓
GlobalState
    ↓
SolidPhases / GasPhases
    ↓
Reaction Models
    ↓
Updated Solid and Gas Phases

Tasarım Amacı

Bu mimarinin hedefleri:

Gerçek klinker faz oluşumunu takip etmek
Stokiyometrik kütle korunumu sağlamak
Dijital ikiz içerisinde fiziksel kimyasal durum oluşturmak
MPC ve optimizasyon algoritmalarına doğru state sağlamak
Gelecekte kinetik parametre kalibrasyonu yapabilmek
AI tabanlı surrogate modeller için temiz veri üretmek