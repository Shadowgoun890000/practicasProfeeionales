from shiny import App

from app.config import MODEL_FILE, MODEL_NAME
from app.ui import create_app_ui
from app.server import create_server
from data.loader import load_data
from data.features import add_time_features
from ml.model_loader import load_model_if_exists

df = load_data()
df = add_time_features(df)

artifact = load_model_if_exists(MODEL_FILE)

app_ui = create_app_ui(df)
server = create_server(df, model=artifact, model_name=MODEL_NAME)

app = App(app_ui, server)