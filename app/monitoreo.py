import argparse
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from app.config import ANIO_CORTE, CAT, NUM, RUTA_MODELO, RUTA_REPORTES
from app.datos import consultar, leer_hechos
from app.features import preparar

EPS = 1e-4


def psi_numerico(base, nuevo, cortes=10):
    bordes = np.unique(np.quantile(base, np.linspace(0, 1, cortes + 1)))
    bordes[0], bordes[-1] = -np.inf, np.inf
    p = np.histogram(base, bins=bordes)[0] / len(base) + EPS
    q = np.histogram(nuevo, bins=bordes)[0] / len(nuevo) + EPS
    return float(np.sum((q - p) * np.log(q / p)))


def psi_categorico(base, nuevo, minimo=0.02):
    # Las categorías raras se juntan en OTRO: con pocas filas darían falsas alertas
    frecuentes = base.value_counts(normalize=True)
    frecuentes = set(frecuentes[frecuentes >= minimo].index)
    base = base.where(base.isin(frecuentes), "OTRO")
    nuevo = nuevo.where(nuevo.isin(frecuentes), "OTRO")
    categorias = sorted(set(base) | set(nuevo))
    p = base.value_counts(normalize=True).reindex(categorias, fill_value=0) + EPS
    q = nuevo.value_counts(normalize=True).reindex(categorias, fill_value=0) + EPS
    return float(np.sum((q - p) * np.log(q / p)))


def estado(psi):
    if psi < 0.1:
        return "estable"
    if psi < 0.25:
        return "atencion"
    return "ALERTA"


def leer_predicciones(ultimas):
    tabla = consultar(f"SELECT entrada, prob FROM predicciones ORDER BY rowid DESC LIMIT {int(ultimas)}")
    entradas = pd.DataFrame([json.loads(texto) for texto in tabla["entrada"]])
    return entradas[NUM + CAT], tabla["prob"]


def deriva_de_entradas(base, nuevo):
    filas = []
    for columna in NUM:
        filas.append((columna, psi_numerico(base[columna].dropna(), nuevo[columna])))
    for columna in CAT:
        filas.append((columna, psi_categorico(base[columna], nuevo[columna])))
    return pd.DataFrame(filas, columns=["variable", "psi"]).assign(
        estado=lambda t: t["psi"].map(estado)
    )


def rendimiento_por_anio():
    modelo = joblib.load(RUTA_MODELO)
    X, y, anio = preparar(leer_hechos())
    filas = []
    for a in sorted(anio.unique()):
        if a <= ANIO_CORTE:
            continue
        mascara = anio == a
        prob = modelo.predict_proba(X[mascara])[:, 1]
        filas.append(
            (int(a), int(mascara.sum()), round(float(y[mascara].mean()), 4),
             round(average_precision_score(y[mascara], prob), 4))
        )
    return pd.DataFrame(filas, columns=["anio", "filas", "prevalencia", "pr_auc"])


def main(ultimas):
    X, _, anio = preparar(leer_hechos())
    base = X[anio <= ANIO_CORTE]
    nuevo, prob = leer_predicciones(ultimas)

    deriva = deriva_de_entradas(base, nuevo)
    rendimiento = rendimiento_por_anio()
    prob_base = joblib.load(RUTA_MODELO).predict_proba(base)[:, 1].mean()

    lineas = [
        f"# Reporte de monitoreo ({datetime.now():%Y-%m-%d %H:%M})",
        "",
        f"Se comparan las últimas {len(nuevo)} predicciones contra el período de entrenamiento.",
        "",
        "## Deriva de las entradas (PSI)",
        "",
        "Menos de 0,10 es estable, entre 0,10 y 0,25 pide atención y más de 0,25 es alerta.",
        "",
        deriva.round(3).to_markdown(index=False),
        "",
        f"Probabilidad media predicha: {prob.mean():.3f} contra {prob_base:.3f} en el entrenamiento.",
        "",
        "## Rendimiento del modelo por año (con datos reales)",
        "",
        rendimiento.to_markdown(index=False),
    ]
    texto = "\n".join(lineas)
    RUTA_REPORTES.mkdir(exist_ok=True)
    (RUTA_REPORTES / "reporte.md").write_text(texto, encoding="utf-8")
    print(texto)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reporte de monitoreo")
    parser.add_argument("--ultimas", type=int, default=300)
    main(parser.parse_args().ultimas)