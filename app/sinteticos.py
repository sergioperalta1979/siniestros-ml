import argparse
import json

import numpy as np
import pandas as pd
import requests

from app.config import ANIO_CORTE, CAT, NUM
from app.datos import leer_hechos
from app.features import preparar


def generar(n, deriva=False, semilla=0):
    """Siniestros sintéticos. Cada columna se remuestrea por separado desde
    el período de entrenamiento: conserva la distribución de cada variable,
    no la relación entre ellas. Con deriva=True cambia la de hora y modo."""
    rng = np.random.default_rng(semilla)
    X, _, anio = preparar(leer_hechos())
    base = X[anio <= ANIO_CORTE]
    sintetico = pd.DataFrame(
        {columna: rng.choice(base[columna].dropna().to_numpy(), size=n) for columna in NUM + CAT}
    )
    if deriva:
        de_noche = rng.random(n) < 0.6
        sintetico.loc[de_noche, "hora"] = rng.choice([22, 23, 0, 1, 2, 3, 4], size=de_noche.sum())
        en_moto = rng.random(n) < 0.5
        sintetico.loc[en_moto, "modo"] = "MOTO"
    return sintetico


def enviar(df, url="http://127.0.0.1:8000/predict"):
    filas = json.loads(df.to_json(orient="records"))
    for fila in filas:
        requests.post(url, json=fila, timeout=10).raise_for_status()
    return len(filas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manda siniestros sintéticos a la API")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--deriva", action="store_true")
    parser.add_argument("--semilla", type=int, default=0)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    args = parser.parse_args()
    datos = generar(args.n, deriva=args.deriva, semilla=args.semilla)
    print("Enviados:", enviar(datos, args.url))