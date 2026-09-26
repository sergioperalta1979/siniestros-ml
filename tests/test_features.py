import pandas as pd

from app.features import preparar


def test_preparar_marca_graves_y_limpia_categorias():
    df = pd.DataFrame(
        {
            "fecha_siniestro": ["2024-03-04", "2019-01-01"],
            "hora_siniestro": ["23:10:00", None],
            "comuna_siniestro": [None, "Comuna 1"],
            "tipo_de_via_siniestro": ["avenida ", "CALLE"],
            "participantes_siniestro": ["MOTO-AUTO", "SD-SD"],
            "modo_desplazamiento_victima": ["MOTO", "SD"],
            "contraparte_siniestro": ["AUTO", "SD"],
            "gravedad_siniestro": ["GRAVE", "LEVE"],
        }
    )
    X, y, anio = preparar(df)
    assert list(y) == [1, 0]
    assert list(anio) == [2024, 2019]
    assert X.loc[0, "comuna"] == "SD"
    assert X.loc[0, "tipo_de_via"] == "AVENIDA"
    assert X.loc[0, "hora"] == 23
    assert pd.isna(X.loc[1, "hora"])