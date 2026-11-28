import numpy as np
import pytest

from cobro_vision.clasificacion.svm import ClasificadorSVM
from cobro_vision.dominio.errores import ModeloIncompatible


class PipelineCalibradoPrueba:

    classes_ = np.array(["a", "b", "c"])
    n_features_in_ = 2

    def __init__(self, probabilidades):
        self.probabilidades = np.asarray(
            probabilidades, dtype=float
        )

    def predict_proba(self, vectores):
        return self.probabilidades

    def decision_function(self, vectores):
        raise AssertionError(
            "Un modelo calibrado no debe usar decision_function."
        )


class PipelineTradicionalPrueba:

    classes_ = np.array(["a", "b", "c"])
    n_features_in_ = 2

    def decision_function(self, vectores):
        return np.array([[2.1, 1.2, 0.1]])


def crear_calibrado(probabilidades, umbral=0.5):

    return ClasificadorSVM(
        PipelineCalibradoPrueba(probabilidades),
        ["a", "b", "c"],
        umbral,
        {"calibracion": {"metodo": "sigmoid"}}
    )


def test_calibrado_alta_confianza():

    modelo = crear_calibrado([[0.8, 0.15, 0.05]])

    resultado = modelo.predecir_lote(
        np.array([[1.0, 2.0]])
    )[0]

    assert resultado.id_clase == "a"
    assert resultado.score == pytest.approx(0.8)
    assert not resultado.requiere_revision


def test_calibrado_baja_confianza():

    modelo = crear_calibrado([[0.35, 0.4, 0.25]])

    resultado = modelo.predecir_lote(
        np.array([[1.0, 2.0]])
    )[0]

    assert resultado.id_clase == "b"
    assert resultado.score == pytest.approx(0.4)
    assert resultado.requiere_revision


def test_calibrado_umbral_exacto():

    modelo = crear_calibrado([[0.5, 0.3, 0.2]])

    resultado = modelo.predecir_lote(
        np.array([[1.0, 2.0]])
    )[0]

    assert not resultado.requiere_revision


def test_calibrado_lote():

    modelo = crear_calibrado([
        [0.8, 0.1, 0.1],
        [0.1, 0.2, 0.7]
    ])

    resultados = modelo.predecir_lote(
        np.array([[1.0, 2.0], [3.0, 4.0]])
    )

    assert len(resultados) == 2
    assert resultados[0].id_clase == "a"
    assert resultados[1].id_clase == "c"


@pytest.mark.parametrize("probabilidades", [
    [[0.8, 0.5, 0.1]],
    [[-0.1, 0.5, 0.6]],
    [[np.nan, 0.5, 0.5]],
    [[0.8, 0.2]],
    [[1.2, 0.0, -0.2]]
])
def test_probabilidades_invalidas(probabilidades):

    modelo = crear_calibrado(probabilidades)

    with pytest.raises(ModeloIncompatible):

        modelo.predecir_lote(
            np.array([[1.0, 2.0]])
        )


def test_calibrado_lote_vacio():

    modelo = crear_calibrado([])

    assert modelo.predecir_lote(
        np.empty((0, 2))
    ) == []


def test_modelo_tradicional_conservado():

    modelo = ClasificadorSVM(
        PipelineTradicionalPrueba(),
        ["a", "b", "c"],
        0.5,
        {}
    )

    resultado = modelo.predecir_lote(
        np.array([[1.0, 2.0]])
    )[0]

    assert resultado.id_clase == "a"
    assert resultado.score == pytest.approx(0.9)
    assert not resultado.requiere_revision

