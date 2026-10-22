"""Contrato y determinismo del clasificador de demostración."""

import numpy as np
import pytest

from cobro_vision.clasificacion.base import Clasificador
from cobro_vision.clasificacion.falso import ClasificadorFalso


def test_contrato_abstracto_y_prediccion_individual() -> None:
    """La base es abstracta y predecir coincide con el lote de una fila."""
    with pytest.raises(TypeError):
        Clasificador()
    clasificador = ClasificadorFalso(["pollo", "fondo"], semilla=42)
    vector = np.array([15, 20, 25], dtype=np.float32)
    assert clasificador.predecir(vector) == clasificador.predecir_lote(vector[None])[0]
    with pytest.raises(ValueError, match="dimensión"):
        clasificador.predecir(vector[None])


def test_falso_independiente_del_orden_y_tamano_del_lote() -> None:
    """El tono y la semilla determinan cada clase sin consumir azar entre llamadas."""
    clases = ["arroz_amarillo", "ensalada", "pollo"]
    vectores = np.array([[0], [1], [2], [3]], dtype=np.float32)
    clasificador = ClasificadorFalso(clases, semilla=42)
    esperado = clasificador.predecir_lote(vectores)
    assert [r.id_clase for r in esperado] == [clases[i] for i in (0, 1, 2, 0)]
    assert clasificador.predecir_lote(vectores) == esperado
    assert ClasificadorFalso(clases, 42).predecir_lote(vectores) == esperado
    assert ClasificadorFalso(clases, 43).predecir_lote(vectores) != esperado
    assert clasificador.predecir_lote(vectores[::-1]) == esperado[::-1]
    assert [clasificador.predecir(v) for v in vectores] == esperado
    assert all(r.score == 10 and not r.requiere_revision for r in esperado)
    assert all(
        r.alternativa in clases and r.alternativa != r.id_clase for r in esperado
    )
    assert clasificador.predecir_lote(np.empty((0, 1), dtype=np.float32)) == []


def test_falso_usa_media_h_o_histograma_y_soporta_una_clase() -> None:
    """La posición de H depende de los nombres, no del orden de extractores."""
    clasificador = ClasificadorFalso(["pollo", "fondo"], semilla=0)
    clasificador.nombres_caracteristicas = ["lbp_00", "hsv_h_media"]
    vector = np.array([999, 1 / 180], dtype=np.float32)
    assert clasificador.predecir(vector).id_clase == "fondo"
    clasificador.nombres_caracteristicas = ["hsv_h_00", "hsv_h_01"]
    assert clasificador.predecir(np.array([1, 0])).id_clase == "fondo"  # H=45
    assert clasificador.predecir(np.array([0, 0])).id_clase == "pollo"
    assert ClasificadorFalso(["pollo"]).predecir(np.zeros(1)).alternativa is None


@pytest.mark.parametrize("clases", [[], ["pollo", "pollo"]])
def test_falso_rechaza_clases_invalidas(clases: list[str]) -> None:
    """No es posible construir predicciones sin clases o con clases duplicadas."""
    with pytest.raises(ValueError):
        ClasificadorFalso(clases)


@pytest.mark.parametrize(
    "vectores", [np.zeros(2), np.array([[np.nan]]), np.array([[np.inf]])]
)
def test_falso_rechaza_matrices_invalidas(vectores: np.ndarray) -> None:
    """Los lotes deben ser bidimensionales y finitos."""
    with pytest.raises(ValueError):
        ClasificadorFalso(["pollo"]).predecir_lote(vectores)
