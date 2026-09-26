import pytest


@pytest.fixture(autouse=True)
def base_temporal(tmp_path, monkeypatch):
    # Las pruebas escriben en una base temporal, no en la de verdad
    monkeypatch.setattr("app.api.RUTA_DB", tmp_path / "prueba.db")