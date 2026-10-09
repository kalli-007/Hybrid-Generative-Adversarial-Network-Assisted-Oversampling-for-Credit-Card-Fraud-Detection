from pathlib import Path

from flask import (Blueprint, current_app, flash, redirect, render_template,
                   request, url_for)
from werkzeug.utils import secure_filename

from .db import execute, query
from .decorators import admin_required
from .ml_engine import (feature_ranking, inspect_dataset, model_ready,
                        preprocessing_summary, train_all)


admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


def latest_dataset():
    return query("SELECT * FROM datasets ORDER BY id DESC LIMIT 1", one=True)


@admin_bp.route("/dashboard")
@admin_required
def dashboard():
    stats = {
        "users": query("SELECT COUNT(*) AS c FROM users WHERE role='user'", one=True)["c"],
        "predictions": query("SELECT COUNT(*) AS c FROM predictions", one=True)["c"],
        "datasets": query("SELECT COUNT(*) AS c FROM datasets", one=True)["c"],
        "models": query("SELECT COUNT(*) AS c FROM training_results", one=True)["c"],
    }
    recent = query("SELECT * FROM training_results ORDER BY id DESC LIMIT 5")
    return render_template("admin/dashboard.html", stats=stats, recent=recent,
                           dataset=latest_dataset(), ready=model_ready(current_app.config["MODEL_FOLDER"]))


@admin_bp.route("/dataset", methods=["GET", "POST"])
@admin_required
def dataset():
    info = None
    if request.method == "POST":
        upload = request.files.get("dataset")
        if not upload or not upload.filename.lower().endswith(".csv"):
            flash("Please upload a valid CSV dataset.", "danger")
        else:
            name = secure_filename(upload.filename)
            path = Path(current_app.config["UPLOAD_FOLDER"]) / f"training_{name}"
            upload.save(path)
            try:
                info = inspect_dataset(path)
                size_kb = round(path.stat().st_size / 1024, 2)
                execute(
                    "INSERT INTO datasets(filename,stored_path,records_count,columns_count,size_kb,fraud_count,legitimate_count) "
                    "VALUES(%s,%s,%s,%s,%s,%s,%s)",
                    (name, str(path), info["records"], info["columns"], size_kb,
                     info["fraud"], info["legitimate"]),
                )
                flash("Dataset uploaded and validated successfully.", "success")
            except Exception as exc:
                path.unlink(missing_ok=True)
                flash(str(exc), "danger")
    data = latest_dataset()
    if data:
        try:
            info = inspect_dataset(data["stored_path"])
        except Exception:
            info = None
    return render_template("admin/dataset.html", dataset=data, info=info)


@admin_bp.route("/preprocessing")
@admin_required
def preprocessing():
    data = latest_dataset()
    if not data:
        flash("Upload a dataset first.", "warning")
        return redirect(url_for("admin.dataset"))
    summary = preprocessing_summary(data["stored_path"])
    return render_template("admin/preprocessing.html", dataset=data, summary=summary)


@admin_bp.route("/features")
@admin_required
def features():
    data = latest_dataset()
    if not data:
        flash("Upload a dataset first.", "warning")
        return redirect(url_for("admin.dataset"))
    ranking = feature_ranking(data["stored_path"])
    return render_template("admin/features.html", dataset=data, ranking=ranking)


@admin_bp.route("/training", methods=["GET", "POST"])
@admin_required
def training():
    data = latest_dataset()
    summary = None
    if request.method == "POST":
        if not data:
            flash("Upload a dataset first.", "warning")
            return redirect(url_for("admin.dataset"))
        try:
            generated = Path(current_app.static_folder) / "generated"
            results, summary = train_all(data["stored_path"], current_app.config["MODEL_FOLDER"], generated)
            execute("DELETE FROM training_results")
            for item in results:
                execute(
                    "INSERT INTO training_results(algorithm,method_type,accuracy,precision_score,recall_score,f1_score,roc_auc,"
                    "confusion_matrix_json,confusion_matrix_image,backend) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (item["algorithm"], item["method_type"], item["accuracy"], item["precision"],
                     item["recall"], item["f1"], item["roc_auc"], str(item["confusion"]),
                     item["confusion_image"], item["backend"]),
                )
            flash("All existing and proposed algorithms were trained successfully.", "success")
        except Exception as exc:
            flash(f"Training failed: {exc}", "danger")
    results = query("SELECT * FROM training_results ORDER BY id")
    return render_template("admin/training.html", dataset=data, results=results, summary=summary)


@admin_bp.route("/comparison")
@admin_required
def comparison():
    results = query("SELECT * FROM training_results ORDER BY id")
    best = max(results, key=lambda x: float(x["f1_score"])) if results else None
    proposed = next((x for x in results if x["method_type"] == "Proposed"), None)
    existing = max((x for x in results if x["method_type"] == "Existing"),
                   key=lambda x: float(x["f1_score"]), default=None)
    improvement = {}
    if proposed and existing:
        for key in ("accuracy", "precision_score", "recall_score", "f1_score", "roc_auc"):
            improvement[key] = round((float(proposed[key]) - float(existing[key])) * 100, 2)
    return render_template("admin/comparison.html", results=results, best=best,
                           proposed=proposed, existing=existing, improvement=improvement)


@admin_bp.route("/users")
@admin_required
def users():
    rows = query("SELECT u.*, COUNT(p.id) AS prediction_count FROM users u LEFT JOIN predictions p ON p.user_id=u.id "
                 "WHERE u.role='user' GROUP BY u.id ORDER BY u.id DESC")
    return render_template("admin/users.html", users=rows)


@admin_bp.route("/predictions")
@admin_required
def predictions():
    rows = query("SELECT p.*,u.name,u.email FROM predictions p JOIN users u ON u.id=p.user_id ORDER BY p.id DESC")
    return render_template("admin/predictions.html", predictions=rows)

