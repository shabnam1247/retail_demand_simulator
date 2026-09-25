from flask import Flask
from pathlib import Path


def create_app():

    app = Flask(__name__)

    BASE_DIR = Path(__file__).resolve().parent.parent

    app.config["BASE_DIR"] = BASE_DIR
    app.config["DATA_DIR"] = BASE_DIR / "data"
    app.config["MODEL_DIR"] = BASE_DIR / "models"

    from app.routes import bp

    app.register_blueprint(bp)

    return app