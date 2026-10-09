from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from .db import execute, query


auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/")
def index():
    return render_template("index.html")


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name or not email or len(password) < 6:
            flash("Enter a valid name, email and a password of at least 6 characters.", "danger")
        elif query("SELECT id FROM users WHERE email=%s", (email,), one=True):
            flash("An account with this email already exists.", "warning")
        else:
            execute(
                "INSERT INTO users(name,email,password_hash,role) VALUES(%s,%s,%s,'user')",
                (name, email, generate_password_hash(password)),
            )
            flash("Registration successful. Please log in.", "success")
            return redirect(url_for("auth.login"))
    return render_template("auth/register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = query("SELECT * FROM users WHERE email=%s", (email,), one=True)
        if not user or not user["active"] or not check_password_hash(user["password_hash"], password):
            flash("Invalid credentials or inactive account.", "danger")
        else:
            session.clear()
            session.update(user_id=user["id"], name=user["name"], role=user["role"])
            destination = "admin.dashboard" if user["role"] == "admin" else "user.dashboard"
            return redirect(url_for(destination))
    return render_template("auth/login.html")


@auth_bp.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.index"))

