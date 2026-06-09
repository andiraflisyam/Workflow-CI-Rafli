# Workflow-CI — Heart Disease MLflow Project

Repository ini berisi workflow CI untuk melatih ulang model machine learning secara otomatis menggunakan **MLflow Project** dan **GitHub Actions**.

## Struktur Folder

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions workflow
├── MLProject/
│   ├── modelling.py            # Script pelatihan utama
│   ├── conda.yaml              # Environment dependencies
│   ├── MLProject               # Konfigurasi MLflow Project
│   └── heart_disease_preprocessing/
│       ├── heart_disease_preprocessing_train.csv
│       └── heart_disease_preprocessing_test.csv
└── README.md
```

## Cara Menjalankan

### Lokal

```bash
# Install dependencies
pip install mlflow==2.19.0 scikit-learn pandas numpy matplotlib seaborn

# Jalankan langsung
cd MLProject
python modelling.py

# Jalankan via MLflow Project
mlflow run MLProject --env-manager=local

# Jalankan dengan custom hyperparameter
mlflow run MLProject --env-manager=local -P n_estimators=200 -P max_depth=15
```

### GitHub Actions

Workflow CI akan otomatis terpantik ketika:
- **Push** ke branch `main`
- **Pull Request** ke branch `main`
- **Manual trigger** melalui tab Actions di GitHub

## Trigger CI

Untuk memantik workflow secara manual:
1. Buka repository di GitHub
2. Klik tab **Actions**
3. Pilih **CI - Heart Disease MLflow Training**
4. Klik **Run workflow**

## Artefak

Setiap run akan menghasilkan:
- `confusion_matrix.png` — Confusion matrix model
- `feature_importance.png` — Feature importance RandomForest
- `classification_report.txt` — Laporan klasifikasi lengkap
- `mlruns/` — Artefak MLflow tracking

Artefak dapat diunduh dari tab **Actions → Run → Artifacts**.
