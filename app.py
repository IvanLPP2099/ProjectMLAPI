from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from statsmodels.tsa.ar_model import AutoRegResults
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = FastAPI(title="API de Modelos de Predicción (UCV 2025)")

# --- Cargar modelos entrenados ---
modelos_prophet = joblib.load("trained_prophet_model.pkl")
modelos_sarimax = joblib.load("trained_sarimax_model.pkl")
modelos_ar = joblib.load("trained_ar_model.pkl")
modelos_ma = joblib.load("trained_ma_model.pkl")
modelos_lstm = joblib.load("trained_lstm_model.pkl")

# --- Entrada esperada ---
class InputData(BaseModel):
    modelo: str  # "Prophet", "SARIMAX", "AR", "MA", "LSTM"
    id: str
    target: str
    fechas: list  # fechas a predecir (opcional)
    ultimos_valores: list = None  # requerido para LSTM o MA


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

    # --- SARIMAX ---
    elif modelo == "sarimax":
        if key not in modelos_sarimax:
            return {"error": f"Modelo {key} no encontrado en SARIMAX"}
        model = modelos_sarimax[key]
        pred = model.predict(start=len(model.data.endog),
                             end=len(model.data.endog) + len(data.fechas) - 1)
        return {
            "modelo": "SARIMAX",
            "id": data.id,
            "target": data.target,
            "predicciones": pred.tolist()
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

    # --- LSTM ---
    elif modelo == "lstm":
        if key not in modelos_lstm:
            return {"error": f"Modelo {key} no encontrado en LSTM"}
        meta = modelos_lstm[key]
        model_path = meta["model_path"]
        scaler = meta["scaler"]
        n_lags = meta["n_lags"]

        if data.ultimos_valores is None or len(data.ultimos_valores) != n_lags:
            return {"error": f"Debe proporcionar {n_lags} valores en 'ultimos_valores'"}

        model = tf.keras.models.load_model(model_path)
        seq = np.array(data.ultimos_valores).reshape(-1, 1)
        seq_scaled = scaler.transform(seq)
        last_seq = seq_scaled[-n_lags:]

        preds = []
        for _ in range(len(data.fechas)):
            x_input = last_seq.reshape(1, n_lags, 1)
            yhat = model.predict(x_input, verbose=0)
            preds.append(yhat[0][0])
            last_seq = np.append(last_seq[1:], yhat)

        preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten().tolist()
        return {
            "modelo": "LSTM",
            "id": data.id,
            "target": data.target,
            "predicciones": preds_inv
        }

    else:
        return {"error": f"Modelo '{data.modelo}' no reconocido"}
