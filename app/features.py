import pandas as pd

from app.config import CAT, NUM


def preparar(df):
    """Devuelve X (variables), y (1 si fue grave o mortal) y el año de cada fila."""
    fecha = pd.to_datetime(df["fecha_siniestro"])
    hora = pd.to_datetime(df["hora_siniestro"], format="%H:%M:%S", errors="coerce").dt.hour

    X = pd.DataFrame(
        {
            "hora": hora,
            "dia_semana": fecha.dt.dayofweek,
            "mes": fecha.dt.month,
            "comuna": df["comuna_siniestro"],
            "tipo_de_via": df["tipo_de_via_siniestro"],
            "participantes": df["participantes_siniestro"],
            "modo": df["modo_desplazamiento_victima"],
            "contraparte": df["contraparte_siniestro"],
        }
    )
    for columna in CAT:
        X[columna] = X[columna].fillna("SD").astype(str).str.strip().str.upper()

    y = df["gravedad_siniestro"].isin(["GRAVE", "MORTAL"]).astype(int)
    return X[NUM + CAT], y, fecha.dt.year