from fastapi.testclient import TestClient

from app.api import app

EJEMPLO = {
    "hora": 23,
    "dia_semana": 5,
    "mes": 11,
    "comuna": "Comuna 1",
    "tipo_de_via": "AVENIDA",
    "participantes": "MOTO-AUTO",
    "modo": "MOTO",
    "contraparte": "AUTO",
}


def test_health():
    with TestClient(app) as cliente:
        respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json()["estado"] == "ok"


def test_predict_devuelve_probabilidad():
    with TestClient(app) as cliente:
        respuesta = cliente.post("/predict", json=EJEMPLO)
    assert respuesta.status_code == 200
    datos = respuesta.json()
    assert 0 <= datos["prob_grave_o_mortal"] <= 1
    assert isinstance(datos["grave_o_mortal"], bool)


def test_predict_rechaza_hora_invalida():
    malo = {**EJEMPLO, "hora": 30}
    with TestClient(app) as cliente:
        respuesta = cliente.post("/predict", json=malo)
    assert respuesta.status_code == 422