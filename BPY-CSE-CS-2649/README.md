# Hybrid GAN-Assisted Credit Card Fraud Detection

This project is a Flask-based academic fraud detection system that compares traditional machine learning models with a proposed Hybrid GAN-assisted oversampling approach for credit card fraud classification.

The application exposes a web interface for:

- admin dataset upload and preprocessing
- feature ranking and model training
- comparing model performance metrics
- user-level single-transaction predictions
- CSV batch fraud analysis with downloadable result files

## Overview

The system is designed for research and coursework use. It trains and compares several models on fraud data, including:

- Logistic Regression
- Decision Tree
- Random Forest + SMOTE
- Gradient Boosting + SMOTE
- Proposed model: Hybrid GAN + Gradient Boosting

It uses a MySQL database to store users, uploaded datasets, training results, and prediction history.

## Tech stack

- Python 3.10+
- Flask 3
- MySQL 8
- PyMySQL
- pandas, numpy, scikit-learn
- matplotlib
- TensorFlow/Keras
- joblib
- python-dotenv

## Project structure

```text
BPY-CSE-CS-2649/
├── app/
│   ├── __init__.py
│   ├── admin.py
│   ├── advisor.py
│   ├── auth.py
│   ├── db.py
│   ├── decorators.py
│   ├── ml_engine.py
│   ├── user.py
│   ├── static/
│   └── templates/
├── model_artifacts/
├── sample_data/
├── uploads/
├── config.py
├── generate_dataset.py
├── install_windows.bat
├── README.md
├── requirements.txt
├── run.py
├── run_windows.bat
├── schema.sql
├── VALIDATED_METRICS.md
└── .env
```

## Key functionality

### Administrator features

- secure admin login
- upload and validate CSV datasets
- inspect dataset statistics and class distribution
- review preprocessing details and feature rankings
- train classical and GAN-assisted fraud models
- compare trained models using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices
- view users and prediction history

### User features

- registration and login
- manual single-transaction prediction
- fraud probability and confidence scores
- AI-generated risk suggestions
- batch CSV upload for up to 10,000 transactions
- risk distribution charts and summary narrative
- downloadable prediction results
- personal prediction history

## Database and configuration

The app reads its configuration from `config.py` and optional environment variables via `.env`.

Default settings include:

- DB host: `localhost`
- DB port: `3306`
- DB user: `root`
- DB password: `root`
- DB name: `credit_fraud_gan`

For local setup, create or update `.env` with values such as:

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=credit_fraud_gan
SECRET_KEY=your-secret-key
```

## Setup

### Prerequisites

- Python 3.10+
- MySQL 8 installed and running
- virtual environment support

### Windows quick start

1. Open PowerShell in the project directory.
2. Create a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Create or update `.env` with your MySQL credentials.
5. Start the app:

```powershell
python run.py
```

6. Open:

```text
http://127.0.0.1:5000
```

The app initializes the database and creates the default admin account automatically when it starts.

### Automated Windows setup

You can also use the bundled setup scripts:

```powershell
install_windows.bat
run_windows.bat
```

## Default admin account

- Email: `admin@fraudshield.ai`
- Password: `admin123`

## First-run workflow

1. Sign in as the administrator.
2. Upload `sample_data/credit_card_training.csv` from the admin dataset page.
3. Review preprocessing and feature-ranking results.
4. Train the models from the training panel.
5. Register a standard user account.
6. Use either single prediction or batch CSV analysis to evaluate new transactions.

## Dataset notes

The included sample dataset contains numeric features and a binary target column named `Class`. For custom datasets, the application accepts target columns such as:

- `Class`
- `is_fraud`
- `fraud`
- `target`
- `label`

Batch prediction files without a target column are also supported.

## Metrics and model behavior

Model metrics are evaluated on the untouched validation/test split. Results are computed by the application at runtime rather than being hardcoded.

See [VALIDATED_METRICS.md](VALIDATED_METRICS.md) for the benchmark values obtained on the included dataset.

## Notes

- This project is intended as an educational decision-support system.
- It is not a production-grade banking fraud control system.
- The app supports a TensorFlow GAN backend when available and falls back to a lightweight statistical generation path if needed.
- The database schema is created automatically by the app on startup.

## License

This repository does not include a formal license file. Please confirm with the project owner or academic institution before reusing it outside the intended course or research context.
