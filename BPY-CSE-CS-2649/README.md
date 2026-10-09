# Hybrid GAN-Assisted Credit Card Fraud Detection

A complete academic Flask web application for comparing conventional fraud classifiers with a proposed Hybrid GAN-assisted oversampling model. It supports administrator-controlled ML training, single transaction detection with AI suggestions, and uploaded CSV graph analysis.

## Technology

- Python 3.10.11
- Flask
- MySQL 8
- PyMySQL only (no MySQLdb or SQLAlchemy)
- TensorFlow/Keras dense GAN
- scikit-learn, pandas, matplotlib and Chart.js

## Main features

### Administrator

1. Secure administrator login
2. Upload and validate a CSV dataset
3. Display records, columns, file size, fraud count and class distribution
4. Preprocess data and show the 80%/20% stratified division
5. Rank and select features using mutual information
6. Train four existing methods:
   - Logistic Regression
   - Decision Tree
   - Random Forest + SMOTE
   - Gradient Boosting + SMOTE
7. Train the proposed Hybrid GAN + Gradient Boosting method with a validation-selected fraud threshold
8. Display accuracy, precision, recall, F1-score, ROC-AUC and confusion matrices
9. Compare algorithms and calculate measured proposed-model betterment
10. View registered users and their prediction history

### User

1. Register, log in and open dashboard
2. Enter one transaction manually
3. View existing result, proposed result, risk, confidence and fraud probability
4. Receive rule-grounded AI security suggestions
5. Upload a CSV with up to 10,000 transactions
6. View fraud distribution, risk-level, probability and model comparison graphs
7. Read an automatic file-level security summary
8. Download the row-by-row prediction CSV
9. Review personal prediction history

## Windows setup

### 1. Requirements

- Install Python 3.10.11 and select **Add Python to PATH**.
- Install and start MySQL Server 8.

### 2. Automatic setup

Double-click `install_windows.bat`. It creates a virtual environment, installs packages, creates `.env`, and regenerates sample datasets.

If your MySQL root user has a password, open `.env` and set:

```env
DB_USER=root
DB_PASSWORD=your_mysql_password
```

### 3. Start the application

Activate the environment and run:

```bat
venv\Scripts\activate
python run.py
```

Open <http://127.0.0.1:5000>.

The application automatically creates the database, all tables and the default administrator account. You may alternatively execute `schema.sql` in MySQL Workbench or phpMyAdmin.

## Administrator login

- Email: `admin@fraudshield.ai`
- Password: `admin123`

## First-use workflow

1. Sign in as administrator.
2. Open **Dataset** and upload `sample_data/credit_card_training.csv`.
3. Review **Preprocessing** and **Feature Selection**.
4. Open **Model Training** and click **Train all algorithms**.
5. Register a normal user or log out and create one.
6. Use **Single Detection** or upload `sample_data/test_transactions.csv` in **File Analysis**.

The included 5,000-record academic dataset contains a rare fraud class with clearly
learnable behavioural differences. On the fixed 80/20 validation split, every included
existing algorithm is expected to exceed 90% measured accuracy, while the proposed model
is designed to exceed 95%. Results are always calculated by the application and are never
hardcoded; results for a different uploaded dataset will naturally vary.

See `VALIDATED_METRICS.md` for the complete measured accuracy, precision, recall,
F1-score and ROC-AUC values obtained with the included dataset.

## Dataset format

The included data contains 12 numeric features and a binary `Class` target. The application also accepts public PCA-style credit-card datasets, provided their target column is named `Class`, `is_fraud`, `fraud`, `target`, or `label`.

For batch testing, the target column is not required. Missing feature columns are filled using training medians, although using the same feature columns as training is strongly recommended.

## GAN fallback

With TensorFlow installed, training uses the included dense GAN. On a low-resource system where TensorFlow cannot load, the project stays runnable using a clearly labelled lightweight statistical generation fallback. The interface always displays which backend was actually used; it does not fabricate model metrics.

## Notes

- Training metrics are computed from the untouched 20% test subset.
- The database column is named `precision_score`, avoiding MySQL keyword/parser conflicts.
- This is an educational decision-support project, not a replacement for a bank's production fraud controls.
