import joblib
from sklearn.metrics import confusion_matrix

from app.config import ANIO_CORTE, RUTA_MODELO
from app.datos import leer_hechos
from app.features import preparar

modelo = joblib.load(RUTA_MODELO)
X, y, anio = preparar(leer_hechos())
X_prueba, y_prueba = X[anio > ANIO_CORTE], y[anio > ANIO_CORTE]
prob = modelo.predict_proba(X_prueba)[:, 1]

print("graves reales en la prueba:", int(y_prueba.sum()), "de", len(y_prueba))
for umbral in [0.3, 0.5, 0.7, 0.9]:
    pred = (prob >= umbral).astype(int)
    vn, fp, fn, vp = confusion_matrix(y_prueba, pred).ravel()
    alarmas = fp + vp
    precision = vp / alarmas if alarmas else 0
    recall = vp / (vp + fn)
    print(f"umbral {umbral}: alarmas={alarmas} aciertos={vp} falsas={fp} perdidos={fn} precision={precision:.3f} recall={recall:.3f}")