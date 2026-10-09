import json
import uuid
from pathlib import Path

import pandas as pd
from flask import (Blueprint, current_app, flash, redirect, render_template,
                   request, send_file, session, url_for)
from werkzeug.utils import secure_filename

from .advisor import ai_suggestions, batch_narrative
from .db import execute, query
from .decorators import login_required
from .ml_engine import DEFAULT_FEATURES, model_ready, predict_frame, predict_single


user_bp = Blueprint("user", __name__, url_prefix="/user")


@user_bp.route("/dashboard")
@login_required
def dashboard():
    stats = query(
        "SELECT COUNT(*) AS total, SUM(input_type='single') AS singles, SUM(input_type='file') AS files, "
        "COALESCE(SUM(fraud_records),0) AS fraud FROM predictions WHERE user_id=%s",
        (session["user_id"],), one=True,
    )
    recent = query("SELECT * FROM predictions WHERE user_id=%s ORDER BY id DESC LIMIT 5", (session["user_id"],))
    return render_template("user/dashboard.html", stats=stats, recent=recent,
                           ready=model_ready(current_app.config["MODEL_FOLDER"]))


@user_bp.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if not model_ready(current_app.config["MODEL_FOLDER"]):
        flash("The model is not trained yet. Ask the administrator to train it.", "warning")
        return redirect(url_for("user.dashboard"))
    if request.method == "POST":
        try:
            values = {name: float(request.form.get(name, 0) or 0) for name in DEFAULT_FEATURES}
            result = predict_single(values, current_app.config["MODEL_FOLDER"])
            risk, suggestions = ai_suggestions(result["probability"], values)
            result["risk"] = risk
            execute(
                "INSERT INTO predictions(user_id,input_type,reference_name,existing_result,proposed_result,fraud_probability,"
                "confidence_score,risk_level,total_records,fraud_records) VALUES(%s,'single',%s,%s,%s,%s,%s,%s,1,%s)",
                (session["user_id"], f"Transaction {uuid.uuid4().hex[:8].upper()}", result["existing_result"],
                 result["proposed_result"], result["probability"], result["confidence"], risk,
                 1 if result["proposed_result"] == "Fraud" else 0),
            )
            return render_template("user/single_result.html", result=result, values=values,
                                   suggestions=suggestions)
        except Exception as exc:
            flash(f"Prediction failed: {exc}", "danger")
    return render_template("user/predict.html", features=DEFAULT_FEATURES)


@user_bp.route("/batch", methods=["GET", "POST"])
@login_required
def batch():
    if not model_ready(current_app.config["MODEL_FOLDER"]):
        flash("The model is not trained yet. Ask the administrator to train it.", "warning")
        return redirect(url_for("user.dashboard"))
    if request.method == "POST":
        upload = request.files.get("test_file")
        if not upload or not upload.filename.lower().endswith(".csv"):
            flash("Please upload a valid CSV file.", "danger")
            return render_template("user/batch.html")
        try:
            token = uuid.uuid4().hex[:10]
            original = secure_filename(upload.filename)
            source = Path(current_app.config["UPLOAD_FOLDER"]) / f"batch_{token}_{original}"
            upload.save(source)
            frame = pd.read_csv(source)
            if frame.empty or len(frame) > 10000:
                raise ValueError("The file must contain between 1 and 10,000 transactions.")
            output, bundle = predict_frame(frame, current_app.config["MODEL_FOLDER"])
            result_path = Path(current_app.config["UPLOAD_FOLDER"]) / f"prediction_{token}.csv"
            output.to_csv(result_path, index=False)
            fraud = int((output["Proposed_Result"] == "Fraud").sum())
            existing_fraud = int((output["Existing_Result"] == "Fraud").sum())
            risk_counts = output["Risk_Level"].value_counts().to_dict()
            amount_col = "Amount" if "Amount" in output.columns else None
            summary = {
                "total": len(output), "fraud": fraud, "legitimate": len(output) - fraud,
                "fraud_rate": fraud / len(output) * 100, "existing_fraud": existing_fraud,
                "risk_counts": {k: int(risk_counts.get(k, 0)) for k in ["Low", "Medium", "High", "Critical"]},
                "average_probability": round(float(output["Proposed_Probability"].mean()) * 100, 2),
                "backend": bundle["backend"], "existing_name": bundle["existing_name"],
            }
            graphs = {
                "risk_labels": list(summary["risk_counts"].keys()),
                "risk_values": list(summary["risk_counts"].values()),
                "probabilities": (output["Proposed_Probability"].head(100) * 100).round(2).tolist(),
                "amounts": output[amount_col].head(100).round(2).tolist() if amount_col else list(range(min(100, len(output)))),
            }
            narrative = batch_narrative(summary)
            execute(
                "INSERT INTO predictions(user_id,input_type,reference_name,existing_result,proposed_result,fraud_probability,"
                "confidence_score,risk_level,total_records,fraud_records,result_file) VALUES(%s,'file',%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (session["user_id"], original, f"{existing_fraud} Fraud", f"{fraud} Fraud",
                 fraud / len(output), float(output["Proposed_Probability"].mean()),
                 "Critical" if risk_counts.get("Critical", 0) else "Batch", len(output), fraud, str(result_path)),
            )
            top = output.sort_values("Proposed_Probability", ascending=False).head(10).to_dict("records")
            return render_template("user/batch_result.html", summary=summary, graphs=graphs,
                                   narrative=narrative, top=top, download=result_path.name)
        except Exception as exc:
            flash(f"File analysis failed: {exc}", "danger")
    return render_template("user/batch.html")


@user_bp.route("/download/<filename>")
@login_required
def download(filename):
    safe = secure_filename(filename)
    if not safe.startswith("prediction_"):
        flash("Invalid download request.", "danger")
        return redirect(url_for("user.history"))
    path = Path(current_app.config["UPLOAD_FOLDER"]) / safe
    if not path.exists():
        flash("Result file is no longer available.", "warning")
        return redirect(url_for("user.history"))
    return send_file(path, as_attachment=True, download_name=safe)


@user_bp.route("/history")
@login_required
def history():
    rows = query("SELECT * FROM predictions WHERE user_id=%s ORDER BY id DESC", (session["user_id"],))
    return render_template("user/history.html", predictions=rows)

