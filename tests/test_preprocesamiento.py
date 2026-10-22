"""Reducción y balance de blancos con imágenes artificiales."""

import cv2
import numpy as np
import pytest

from cobro_vision.preprocesamiento.corrector_color import CorrectorColor
from cobro_vision.preprocesamiento.preprocesador import Preprocesador


@pytest.mark.parametrize(
    "forma,esperada,factor",
    [
        ((900, 1200, 3), (480, 640, 3), 640 / 1200),
        ((1200, 900, 3), (640, 480, 3), 640 / 1200),
        ((100, 160, 3), (100, 160, 3), 1.0),
        ((1, 2000, 3), (1, 640, 3), 640 / 2000),
    ],
)
def test_tamano_escala_y_bgr(
    datos_config: dict,
    forma: tuple[int, int, int],
    esperada: tuple[int, int, int],
    factor: float,
) -> None:
    """Reduce sin ampliar, conserva proporción y no permuta canales."""
    config = datos_config["imagen"]
    config["correccion_color"]["metodo"] = "ninguna"
    imagen = np.full(forma, (25, 100, 180), np.uint8)
    original = imagen.copy()
    preprocesador = Preprocesador(config)
    assert preprocesador.escala(imagen) == pytest.approx(factor)
    salida = preprocesador.procesar(imagen)
    assert salida.shape == esperada
    assert salida.dtype == np.uint8
    np.testing.assert_array_equal(salida[0, 0], (25, 100, 180))
    np.testing.assert_array_equal(imagen, original)
    assert not np.shares_memory(salida, imagen)


@pytest.mark.parametrize("tipo", ["mediana", "gaussiano", "ninguno"])
def test_filtro_configurado(datos_config: dict, tipo: str) -> None:
    """Un impulso distingue el comportamiento de los tres filtros."""
    config = datos_config["imagen"]
    config["filtro"] = {"tipo": tipo, "tamano": 3}
    config["correccion_color"]["metodo"] = "ninguna"
    imagen = np.zeros((11, 11, 3), dtype=np.uint8)
    imagen[5, 5] = 255
    salida = Preprocesador(config).procesar(imagen)
    if tipo == "mediana":
        assert not salida.any()
    elif tipo == "gaussiano":
        assert 0 < salida[5, 5, 0] < 255
        assert salida[4, 5, 0] > 0
    else:
        np.testing.assert_array_equal(salida, imagen)


def test_gray_world_y_limites() -> None:
    """Neutraliza dominantes y acota ganancias sin NaN en canales negros."""
    imagen = np.full((20, 30, 3), (50, 100, 150), np.uint8)
    corregida = CorrectorColor().corregir(imagen)
    np.testing.assert_array_equal(corregida, np.full_like(imagen, 100))
    extrema = np.full_like(imagen, (0, 1, 255))
    salida = CorrectorColor().corregir(extrema)
    np.testing.assert_array_equal(salida[0, 0], (0, 2, 127))
    negra = np.zeros_like(imagen)
    np.testing.assert_array_equal(CorrectorColor().corregir(negra), negra)
    assert salida.dtype == np.uint8


def test_preprocesador_corrige_despues_del_filtro(datos_config: dict) -> None:
    """La corrección configurada forma parte del preprocesamiento."""
    imagen = np.full((10, 10, 3), (50, 100, 150), np.uint8)
    salida = Preprocesador(datos_config["imagen"]).procesar(imagen)
    assert np.all(salida == 100)


def test_tarjeta_gris_solo_usa_roi() -> None:
    """Calcula ganancias en el ROI y las aplica a toda la imagen."""
    imagen = np.full((12, 12, 3), (20, 60, 120), np.uint8)
    imagen[:4, :4] = (50, 100, 150)
    salida = CorrectorColor({"metodo": "tarjeta_gris", "roi": [0, 0, 4, 4]}).corregir(
        imagen
    )
    np.testing.assert_array_equal(salida[0, 0], (100, 100, 100))
    np.testing.assert_array_equal(salida[6, 6], (40, 60, 80))
    with pytest.raises(ValueError, match="ROI"):
        CorrectorColor({"metodo": "tarjeta_gris", "roi": [11, 0, 4, 4]}).corregir(
            imagen
        )


def test_ninguna_y_entrada_invalida(datos_config: dict) -> None:
    """La copia sin corrección conserva sus valores y valida el contrato BGR."""
    imagen = np.full((8, 8, 3), (10, 80, 200), np.uint8)
    salida = CorrectorColor({"metodo": "ninguna"}).corregir(imagen)
    np.testing.assert_array_equal(salida, imagen)
    assert not np.shares_memory(salida, imagen)
    for invalida in (
        np.zeros((4, 4), np.uint8),
        imagen.astype(float),
        np.zeros((0, 3, 3), np.uint8),
    ):
        with pytest.raises(ValueError, match="BGR uint8"):
            Preprocesador(datos_config["imagen"]).procesar(invalida)


def test_redimensionamiento_area(datos_config: dict) -> None:
    """Al reducir un patrón alternado, INTER_AREA promedia los bloques."""
    config = datos_config["imagen"]
    config["lado_mayor"] = 2
    config["filtro"]["tipo"] = "ninguno"
    config["correccion_color"]["metodo"] = "ninguna"
    patron = np.tile(np.array([0, 200], np.uint8), (4, 2))
    imagen = cv2.cvtColor(patron, cv2.COLOR_GRAY2BGR)
    assert np.all(Preprocesador(config).procesar(imagen) == 100)
