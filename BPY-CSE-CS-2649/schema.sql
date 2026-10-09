CREATE DATABASE IF NOT EXISTS credit_fraud_gan CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE credit_fraud_gan;

CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(160) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('admin','user') NOT NULL DEFAULT 'user',
    active TINYINT(1) NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS datasets (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    stored_path VARCHAR(500) NOT NULL,
    records_count INT NOT NULL,
    columns_count INT NOT NULL,
    size_kb DECIMAL(12,2) NOT NULL,
    fraud_count INT DEFAULT 0,
    legitimate_count INT DEFAULT 0,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS training_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    algorithm VARCHAR(160) NOT NULL,
    method_type VARCHAR(40) NOT NULL,
    accuracy DECIMAL(8,5) NOT NULL,
    precision_score DECIMAL(8,5) NOT NULL,
    recall_score DECIMAL(8,5) NOT NULL,
    f1_score DECIMAL(8,5) NOT NULL,
    roc_auc DECIMAL(8,5) NOT NULL,
    confusion_matrix_json TEXT,
    confusion_matrix_image VARCHAR(500),
    backend VARCHAR(100),
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    input_type ENUM('single','file') NOT NULL,
    reference_name VARCHAR(255),
    existing_result VARCHAR(40),
    proposed_result VARCHAR(40),
    fraud_probability DECIMAL(8,5),
    confidence_score DECIMAL(8,5),
    risk_level VARCHAR(30),
    total_records INT DEFAULT 1,
    fraud_records INT DEFAULT 0,
    result_file VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_predictions_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

