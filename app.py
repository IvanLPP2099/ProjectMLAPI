from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.ar_model import AutoRegResults
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = FastAPI(title="API de Modelos de Predicción (UCV 2025)")

# --- Cargar modelos entrenados ---
modelos_prophet = joblib.load("trained_prophet_model.pkl")
modelos_ar = joblib.load("trained_autoregression_model.pkl")
modelos_ma = joblib.load("trained_ma_model.pkl")


# --- Entrada esperada ---
class InputData(BaseModel):
    modelo: str  # "Prophet", "SARIMAX", "AR", "MA", "LSTM"
    id: str
    target: str
    fechas: list  # fechas a predecir (opcional)
    ultimos_valores: list = None  # requerido para LSTM o MA

@app.get("/")
def home():
    return {"message": "API de modelos de predicción funcionando correctamente"}


@app.post("/predict")
def predict(data: InputData):
    modelo = data.modelo.lower()
    key = f"{data.id}_{data.target}"

    # --- Prophet ---
    if modelo == "prophet":
        if key not in modelos_prophet:
            return {"error": f"Modelo {key} no encontrado en Prophet"}
        model = modelos_prophet[key]
        future = pd.DataFrame({"ds": pd.to_datetime(data.fechas)})
        forecast = model.predict(future)
        return {
            "modelo": "Prophet",
            "id": data.id,
            "target": data.target,
            "predicciones": forecast[["ds", "yhat"]].to_dict(orient="records")
        }

    # --- AutoReg ---
    elif modelo == "ar":
        if key not in modelos_ar:
            return {"error": f"Modelo {key} no encontrado en AutoReg"}
        model = modelos_ar[key]
        pred = model.predict(start=len(model.model.endog),
                             end=len(model.model.endog) + len(data.fechas) - 1)
        return {
            "modelo": "AutoReg",
            "id": data.id,
            "target": data.target,
            "predicciones": pred.tolist()
        }

    # --- Media Móvil ---
    elif modelo == "ma":
        if key not in modelos_ma:
            return {"error": f"Modelo {key} no encontrado en MA"}
        meta = modelos_ma[key]
        pred = [meta["last_ma"]] * len(data.fechas)
        return {
            "modelo": "Media Móvil",
            "id": data.id,
            "target": data.target,
            "predicciones": pred
        }

    else:
        return {"error": f"Modelo '{data.modelo}' no reconocido"}
