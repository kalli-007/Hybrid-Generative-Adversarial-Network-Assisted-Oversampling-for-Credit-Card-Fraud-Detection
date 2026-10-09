from flask import Flask
from pathlib import Path

from config import Config
from .db import close_db, init_database


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    for key in ("UPLOAD_FOLDER", "MODEL_FOLDER", "DATASET_FOLDER"):
        Path(app.config[key]).mkdir(parents=True, exist_ok=True)

    from .auth import auth_bp
    from .admin import admin_bp
    from .user import user_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(user_bp)
    app.teardown_appcontext(close_db)

    with app.app_context():
        init_database()

    return app
