import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

RUTA_CSV = RAIZ / "data" / "hechos.csv"
RUTA_DB = Path(os.getenv("SINIESTROS_DB") or RAIZ / "data" / "siniestros.db")
RUTA_MODELO = RAIZ / "modelos" / "modelo.joblib"
RUTA_INFO = RAIZ / "modelos" / "info.json"
RUTA_REPORTES = RAIZ / "reportes"

URL_HECHOS = (
    "https://data.buenosaires.gob.ar/dataset/victimas-siniestros-viales/"
    "resource/40ec993a-00ad-40e5-936f-0a25f8d2c90b/download"
)

CAT = ["comuna", "tipo_de_via", "participantes", "modo", "contraparte"]
NUM = ["hora", "dia_semana", "mes"]

# Los siniestros de 2024 en adelante se reservan para probar el modelo
ANIO_CORTE = 2023