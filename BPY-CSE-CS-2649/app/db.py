import pymysql
from flask import current_app, g
from werkzeug.security import generate_password_hash


def server_connection():
    return pymysql.connect(
        host=current_app.config["DB_HOST"],
        port=current_app.config["DB_PORT"],
        user=current_app.config["DB_USER"],
        password=current_app.config["DB_PASSWORD"],
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def get_db():
    if "db" not in g:
        g.db = pymysql.connect(
            host=current_app.config["DB_HOST"],
            port=current_app.config["DB_PORT"],
            user=current_app.config["DB_USER"],
            password=current_app.config["DB_PASSWORD"],
            database=current_app.config["DB_NAME"],
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )
    return g.db


def query(sql, args=(), one=False):
    with get_db().cursor() as cur:
        cur.execute(sql, args)
        rows = cur.fetchall()
    return (rows[0] if rows else None) if one else rows


def execute(sql, args=()):
    with get_db().cursor() as cur:
        cur.execute(sql, args)
        return cur.lastrowid


def init_database():
    name = current_app.config["DB_NAME"]
    conn = server_connection()
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE DATABASE IF NOT EXISTS `{name}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
    conn.close()

    schema = """
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
    """
    db = get_db()
    with db.cursor() as cur:
        for statement in schema.split(";"):
            if statement.strip():
                cur.execute(statement)

    if not query("SELECT id FROM users WHERE email=%s", ("admin@fraudshield.ai",), one=True):
        execute(
            "INSERT INTO users(name,email,password_hash,role) VALUES(%s,%s,%s,'admin')",
            ("System Administrator", "admin@fraudshield.ai", generate_password_hash("admin123")),
        )


def close_db(_error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()

