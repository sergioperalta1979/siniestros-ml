import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.config import CAT
from app.datos import leer_hechos
from app.entrenar import dividir
from app.features import preparar

X, y, anio = preparar(leer_hechos())
angulo = 2 * np.pi * X["hora"] / 24
X["hora_sen"] = np.sin(angulo)
X["hora_cos"] = np.cos(angulo)
X_ent, X_prueba, y_ent, y_prueba = dividir(X, y, anio)


def evaluar(clasificador, numericas):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numericas),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100, sparse_output=False), CAT),
    ])
    pipeline = Pipeline([("pre", pre), ("modelo", clasificador)]).fit(X_ent, y_ent)
    prob = pipeline.predict_proba(X_prueba)[:, 1]
    return round(average_precision_score(y_prueba, prob), 4), round(roc_auc_score(y_prueba, prob), 4)


normal = ["hora", "dia_semana", "mes"]
ciclica = ["hora_sen", "hora_cos", "dia_semana", "mes"]
for nombre, crear in [
    ("regresion logistica", lambda: LogisticRegression(max_iter=1000, class_weight="balanced")),
    ("gradient boosting", lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced", random_state=0)),
]:
    print(nombre, "| hora normal:", evaluar(crear(), normal), "| hora ciclica:", evaluar(crear(), ciclica))