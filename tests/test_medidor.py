"""Mediciones reproducibles mediante un reloj simulado."""

import pytest

from cobro_vision.controlador.medidor import MedidorTiempos


def test_acumula_milisegundos_incluye_fallos_y_reinicia(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Un fallo conserva el tiempo y resumen devuelve una copia sin efectos."""
    instantes = iter([0, 0.125, 1, 1.250, 2, 2.500])
    monkeypatch.setattr(
        "cobro_vision.controlador.medidor.perf_counter", lambda: next(instantes)
    )
    medidor = MedidorTiempos()
    with medidor.medir("segmentacion"):
        pass
    with medidor.medir("segmentacion"):
        pass
    with pytest.raises(ValueError), medidor.medir("clasificacion"):
        raise ValueError("fallo simulado")
    assert medidor.resumen() == {
        "segmentacion": 375,
        "clasificacion": 500,
        "total": 875,
    }
    copia = medidor.resumen()
    copia["segmentacion"] = 0
    assert medidor.resumen()["segmentacion"] == 375
    medidor.reiniciar()
    assert medidor.resumen() == {"total": 0}
