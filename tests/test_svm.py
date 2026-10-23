"""Decisiones binarias y multiclase, persistencia y compatibilidad."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cobro_vision.clasificacion.svm import ClasificadorSVM
from cobro_vision.dominio.errores import ModeloIncompatible


@pytest.mark.parametrize(
    ("clases", "decisiones", "esperadas", "margenes", "alternativas"),
    [
        (
            ["a", "b"],
            np.array([-2.0, 0.1, 0]),
            ["a", "b", "a"],
            [4, 0.2, 0],
            ["b", "a", "b"],
        ),
        (
            ["a", "b", "c"],
            np.array([[1, 3, 2], [2, 2, 0], [0, 1, 4]]),
            ["b", "a", "c"],
            [1, 0, 3],
            ["c", "b", "b"],
        ),
    ],
)
def test_decisiones_margen_alternativa_y_revision(
    clases: list[str],
    decisiones: np.ndarray,
    esperadas: list[str],
    margenes: list[float],
    alternativas: list[str],
) -> None:
    """El mejor score decide la clase y el empate mantiene el orden estable."""
    pipeline = SimpleNamespace(
        classes_=clases,
        n_features_in_=2,
        decision_function=lambda _: decisiones,
    )
    modelo = ClasificadorSVM(pipeline, clases, 1.0, {})
    resultados = modelo.predecir_lote(np.zeros((3, 2)))
    assert [r.id_clase for r in resultados] == esperadas
    assert [r.alternativa for r in resultados] == alternativas
    assert [r.score for r in resultados] == pytest.approx(margenes)
    assert [r.requiere_revision for r in resultados] == [m < 1 for m in margenes]
    assert modelo.predecir_lote(np.empty((0, 2))) == []
    for vectores in (np.zeros(2), np.zeros((3, 1)), np.full((1, 2), np.nan)):
        with pytest.raises(ValueError, match="dimensión"):
            modelo.predecir_lote(vectores)


def test_cargar_archivo_con_formato_invalido(tmp_path: Path) -> None:
    """Un joblib que no contiene el contrato del modelo falla de forma reconocible."""
    import joblib

    ruta = tmp_path / "invalido.joblib"
    joblib.dump({}, ruta)
    with pytest.raises(ModeloIncompatible):
        ClasificadorSVM.cargar(ruta, SimpleNamespace(nombres=list))
