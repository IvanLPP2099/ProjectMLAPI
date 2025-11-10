
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import gzip
import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

app = FastAPI(title="API Forecast Warehouse")

# --- Cargar modelos entrenados ---
with gzip.open("model_prophet.pkl.gz", "rb") as f:
    modelos_prophet = pickle.load(f)

with gzip.open("model_autoreg.pkl.gz", "rb") as f:
    modelos_ar = pickle.load(f)

with gzip.open("model_sarimax.pkl.gz", "rb") as f:
    modelos_sarimax = pickle.load(f)

# --- Estructura del request ---
class InputData(BaseModel):
    modelo: str
    id: str
    target: str
    fechas: List[str]


@app.get("/")
def home():
    return {"message": " API de predicción lista"}


@app.post("/predict")
def predict(data: InputData):
    modelo = data.modelo.lower()
    key = f"{data.id}_{data.target}"

    # --- Validar fechas ---
    if not data.fechas:
        return {"error": "Debes enviar al menos una fecha en 'fechas'."}

    try:
        fechas_pred = pd.to_datetime(data.fechas)
        fechas_pred = fechas_pred.sort_values()
    except Exception:
        return {"error": "Formato de fechas inválido. Usa 'YYYY-MM-DD'."}

    meses = len(fechas_pred)

    # --- Seleccionar modelo ---
    if modelo == "prophet":
        modelos_dict = modelos_prophet
    elif modelo in "autoregression":
        modelos_dict = modelos_ar
    elif modelo == "sarimax":
        modelos_dict = modelos_sarimax
    else:
        return {"error": f"Modelo '{data.modelo}' no reconocido."}

    if key not in modelos_dict:
        return {"error": f"No se encontró modelo para {key}."}

    # --- Predicción según modelo ---
    predicciones = []
    if modelo == "prophet":
        model = modelos_dict[key]
        future = pd.DataFrame({"ds": fechas_pred})
        forecast = model.predict(future)

        predicciones = pd.DataFrame({
            "fecha": forecast["ds"],
            "yhat": forecast["yhat"].round(3)
        })

    elif modelo in ["autoregression", "ar"]:
        modelo_info = modelos_dict[key]
        res_model = modelo_info["modelo"]
        pred = res_model.predict(
            start=res_model.nobs,
            end=res_model.nobs + meses - 1
        )

        predicciones = pd.DataFrame({
            "fecha": fechas_pred,
            "yhat": np.round(pred.values, 3)
        })

    elif modelo == "sarimax":
        res_model = modelos_dict[key]
        pred = res_model.get_forecast(steps=meses)
        forecast_mean = pred.predicted_mean

        predicciones = pd.DataFrame({
            "fecha": fechas_pred,
            "yhat": np.round(forecast_mean.values, 3)
        })

    # ==========================================
    # RESPUESTA FINAL
    # ==========================================
    return {
        "modelo": modelo.upper(),
        "id": data.id,
        "target": data.target,
        "fechas": [d.strftime("%Y-%m-%d") for d in fechas_pred],
        "predicciones": predicciones.to_dict(orient="records"),
    }
