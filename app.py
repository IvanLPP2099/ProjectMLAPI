from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

app = FastAPI(title="API de Modelos de Predicción (UCV 2025)")

# --- Cargar modelos y métricas ---
modelos_prophet = joblib.load("trained_prophet_model.pkl")
modelos_ar = joblib.load("trained_autoregression_model.pkl")
modelos_ma = joblib.load("trained_ma_model.pkl")
modelos_sarimax = joblib.load("trained_sarimax_model.pkl")

# Resultados históricos
results_prophet = pd.read_csv("results_prophet.csv", sep=";")
results_ar = pd.read_csv("results_autoreg.csv", sep=";")
results_ma = pd.read_csv("results_ma.csv", sep=";")
results_sarimax = pd.read_csv("results_sarimax.csv", sep=";")


class InputData(BaseModel):
    modelo: str
    id: str
    target: str
    meses: int = 3  # número de meses a predecir


@app.get("/")
def home():
    return {"message": "API de modelos de predicción funcionando correctamente"}


@app.post("/predict")
def predict(data: InputData):
    modelo = data.modelo.lower()
    key = f"{data.id}_{data.target}"

    # --- Seleccionar fuente de métricas ---
    if modelo == "prophet":
        results_df = results_prophet
        modelos_dict = modelos_prophet
    elif modelo == "ar":
        results_df = results_ar
        modelos_dict = modelos_ar
    elif modelo == "ma":
        results_df = results_ma
        modelos_dict = modelos_ma
    elif modelo == "sarimax":
        results_df = results_sarimax
        modelos_dict = modelos_sarimax
    else:
        return {"error": f"Modelo '{data.modelo}' no reconocido"}

    # --- Buscar métricas históricas ---
    fila = results_df.query("id == @data.id and target == @data.target")
    metricas = None
    if not fila.empty:
        fila = fila.iloc[0]
        metricas = {
            "mae_test": round(fila["mae_test"], 3),
            "rmse_test": round(fila["rmse_test"], 3),
            "r2_test": round(fila["r2_test"], 3),
            "mae_validate": round(fila["mae_validate"], 3),
            "rmse_validate": round(fila["rmse_validate"], 3),
            "r2_validate": round(fila["r2_validate"], 3)
        }

    # --- Generar predicciones ---
    if modelo == "prophet":
        if key not in modelos_dict:
            return {"error": f"Modelo {key} no encontrado en Prophet"}
        model = modelos_dict[key]

        last_date = model.history["ds"].max()
        fechas_pred = pd.date_range(
            start=last_date + pd.offsets.MonthBegin(),
            periods=data.meses,
            freq="MS"
        )
        future = pd.DataFrame({"ds": fechas_pred})
        forecast = model.predict(future)

        predicciones = [
            {"fecha": d.strftime("%Y-%m-%d"), "yhat": round(y, 3)}
            for d, y in zip(forecast["ds"], forecast["yhat"])
        ]

    elif modelo == "ar":
        if key not in modelos_dict:
            return {"error": f"Modelo {key} no encontrado en AutoReg"}
        model = modelos_dict[key]
        pred = model.predict(
            start=len(model.model.endog),
            end=len(model.model.endog) + data.meses - 1
        )
        predicciones = [
            {"mes": i + 1, "valor": round(float(p), 3)} for i, p in enumerate(pred)
        ]

    elif modelo == "ma":
        if key not in modelos_dict:
            return {"error": f"Modelo {key} no encontrado en MA"}
        meta = modelos_dict[key]
        pred = [meta["last_ma"]] * data.meses
        predicciones = [
            {"mes": i + 1, "valor": round(float(p), 3)} for i, p in enumerate(pred)
        ]

    elif modelo == "sarimax":
        if key not in modelos_dict:
            return {"error": f"Modelo {key} no encontrado en SARIMAX"}

        cfg = modelos_dict[key]

        # Reconstruir serie con últimos valores
        serie = pd.Series(cfg.get("last_train_value", [0]*12))
        model = SARIMAX(serie, order=cfg["order"], seasonal_order=cfg["seasonal_order"])
        res = model.filter(cfg["params"])  # aplica parámetros del modelo entrenado

        forecast = res.get_forecast(steps=data.meses)
        pred = forecast.predicted_mean.tolist()
        predicciones = [
            {"mes": i + 1, "valor": round(float(p), 3)} for i, p in enumerate(pred)
        ]

    else:
        return {"error": f"Modelo '{data.modelo}' no reconocido"}

    # --- Respuesta final ---
    return {
        "modelo": data.modelo.upper(),
        "id": data.id,
        "target": data.target,
        "metricas_historicas": metricas,
        "predicciones": predicciones
    }
