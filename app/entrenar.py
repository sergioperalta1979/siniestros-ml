import json
from datetime import date

import joblib
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.config import ANIO_CORTE, CAT, NUM, RUTA_INFO, RUTA_MODELO
from app.datos import leer_hechos
from app.features import preparar

UMBRAL = 0.5


def armar_preprocesador():
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUM),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=100,
                    sparse_output=False,
                ),
                CAT,
            ),
        ]
    )


def dividir(X, y, anio):
    entrenamiento = anio <= ANIO_CORTE
    return X[entrenamiento], X[~entrenamiento], y[entrenamiento], y[~entrenamiento]


def medir(modelo, X, y):
    prob = modelo.predict_proba(X)[:, 1]
    pred = (prob >= UMBRAL).astype(int)
    return {
        "pr_auc": round(average_precision_score(y, prob), 4),
        "roc_auc": round(roc_auc_score(y, prob), 4),
        "recall": round(recall_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
    }


def candidatos():
    return {
        "azar (prior)": DummyClassifier(strategy="prior"),
        "regresion logistica": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "gradient boosting": HistGradientBoostingClassifier(
    max_iter=200, learning_rate=0.1, class_weight="balanced", random_state=0
),
    }


def main():
    X, y, anio = preparar(leer_hechos())
    X_ent, X_prueba, y_ent, y_prueba = dividir(X, y, anio)
    print(f"Entrenamiento: {len(X_ent)} filas ({y_ent.mean():.1%} graves o mortales)")
    print(f"Prueba: {len(X_prueba)} filas ({y_prueba.mean():.1%} graves o mortales)")

    resultados = {}
    modelos = {}
    for nombre, clasificador in candidatos().items():
        pipeline = Pipeline([("pre", armar_preprocesador()), ("modelo", clasificador)])
        pipeline.fit(X_ent, y_ent)
        resultados[nombre] = medir(pipeline, X_prueba, y_prueba)
        modelos[nombre] = pipeline
        print(nombre, resultados[nombre])

    mejor = max(resultados, key=lambda nombre: resultados[nombre]["pr_auc"])
    print("Mejor por PR-AUC:", mejor)
    RUTA_MODELO.parent.mkdir(exist_ok=True)
    joblib.dump(modelos[mejor], RUTA_MODELO)
    info = {
        "modelo": mejor,
        "fecha_entrenamiento": date.today().isoformat(),
        "umbral": UMBRAL,
        "metricas_prueba": resultados[mejor],
        "sklearn": sklearn.__version__,
    }
    RUTA_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Modelo guardado en", RUTA_MODELO)


if __name__ == "__main__":
    main()