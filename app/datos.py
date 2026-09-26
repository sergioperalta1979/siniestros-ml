import sqlite3

import pandas as pd
import requests

from app.config import RUTA_CSV, RUTA_DB, URL_HECHOS


def descargar(url=URL_HECHOS, destino=RUTA_CSV):
    destino.parent.mkdir(exist_ok=True)
    respuesta = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
    respuesta.raise_for_status()
    destino.write_bytes(respuesta.content)
    return destino

def cargar_a_sqlite(ruta_csv=RUTA_CSV, ruta_db=RUTA_DB):
    df = pd.read_csv(ruta_csv, sep=";", encoding="utf-8-sig", low_memory=False)
    conexion = sqlite3.connect(ruta_db)
    df.to_sql("hechos", conexion, if_exists="replace", index=False)
    conexion.close()
    return len(df)