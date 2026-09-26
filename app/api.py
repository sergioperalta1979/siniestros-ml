import json
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.config import CAT, NUM, RUTA_DB, RUTA_INFO, RUTA_MODELO

estado = {}


class Siniestro(BaseModel):
    hora: int = Field(ge=0, le=23)
    dia_semana: int = Field(ge=0, le=6, description="0 = lunes")
    mes: int = Field(ge=1, le=12)
    comuna: str = "SD"
    tipo_de_via: str = "SD"
    participantes: str = "SD"
    modo: str = "SD"
    contraparte: str = "SD"


class Prediccion(BaseModel):
    prob_grave_o_mortal: float
    grave_o_mortal: bool
    umbral: float
    modelo: str
    

def crear_tabla_predicciones():
    RUTA_DB.parent.mkdir(exist_ok=True)
    conexion = sqlite3.connect(RUTA_DB)
    conexion.execute(
        "CREATE TABLE IF NOT EXISTS predicciones "
        "(fecha TEXT, entrada TEXT, prob REAL)"
    )
    conexion.commit()
    conexion.close()


def registrar(entrada, prob):
    conexion = sqlite3.connect(RUTA_DB)
    conexion.execute(
        "INSERT INTO predicciones VALUES (?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), json.dumps(entrada), prob),
    )
    conexion.commit()
    conexion.close()
    

@asynccontextmanager
async def ciclo_de_vida(app):
    estado["modelo"] = joblib.load(RUTA_MODELO)
    estado["info"] = json.loads(RUTA_INFO.read_text(encoding="utf-8"))
    crear_tabla_predicciones()
    yield
    estado.clear()


app = FastAPI(title="Siniestros viales CABA", lifespan=ciclo_de_vida)


@app.get("/health")
def health():
    return {"estado": "ok", "modelo": estado["info"]["modelo"]}


@app.post("/predict", response_model=Prediccion)
def predict(siniestro: Siniestro):
    entrada = siniestro.model_dump()
    for columna in CAT:
        entrada[columna] = entrada[columna].strip().upper()
    fila = pd.DataFrame([entrada])[NUM + CAT]
    prob = float(estado["modelo"].predict_proba(fila)[0, 1])
    registrar(entrada, prob)
    umbral = estado["info"]["umbral"]
    return Prediccion(
        prob_grave_o_mortal=round(prob, 4),
        grave_o_mortal=prob >= umbral,
        umbral=umbral,
        modelo=estado["info"]["modelo"],
    )