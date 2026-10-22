"""Detección del círculo artificial y rechazo de escenas sin plato."""

import cv2
import numpy as np
import pytest

from cobro_vision.dominio.errores import PlatoNoDetectado
from cobro_vision.preprocesamiento.detector_plato import DetectorPlato


@pytest.mark.parametrize("metodo", ["hough", "umbral", "imagen_completa"])
def test_metodos_sobre_plato_sintetico(
    datos_config: dict, plato_sintetico: tuple[np.ndarray, np.ndarray], metodo: str
) -> None:
    """La máscara ubica el plato circular incluso con colores en su interior."""
    imagen, _ = plato_sintetico
    config = datos_config["plato"]
    config["metodo"] = metodo
    mascara = DetectorPlato(config).detectar(imagen)
    assert mascara.dtype == np.bool_
    assert mascara.shape == imagen.shape[:2]
    if metodo == "imagen_completa":
        assert mascara.all()
    else:
        filas, columnas = np.indices(mascara.shape)
        esperada = (filas - 160) ** 2 + (columnas - 160) ** 2 <= 140**2
        iou = np.count_nonzero(mascara & esperada) / np.count_nonzero(
            mascara | esperada
        )
        assert iou > 0.85
        assert mascara[160, 160] and not mascara[0, 0]


def test_hough_real_sin_respaldo(
    datos_config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hough detecta un círculo limpio y reduce su radio sin recurrir al umbral."""
    imagen = np.full((480, 480, 3), 40, dtype=np.uint8)
    cv2.circle(imagen, (240, 240), 190, (240, 240, 240), -1, cv2.LINE_AA)
    detector = DetectorPlato(datos_config["plato"])

    def respaldo_prohibido(imagen: np.ndarray) -> np.ndarray:
        raise AssertionError("Hough debía localizar este círculo")

    monkeypatch.setattr(detector, "_umbral", respaldo_prohibido)
    mascara = detector.detectar(imagen)
    radio_equivalente = np.sqrt(mascara.sum() / np.pi)
    assert radio_equivalente == pytest.approx(190 * 0.97, abs=4)


def test_respaldo_hough_y_primer_circulo(
    datos_config: dict,
    plato_sintetico: tuple[np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sin círculos usa umbral; con varios toma el primero y contrae su radio."""
    imagen, _ = plato_sintetico
    config = datos_config["plato"]
    monkeypatch.setattr(cv2, "HoughCircles", lambda *args, **kwargs: None)
    respaldo = DetectorPlato(config).detectar(imagen)
    np.testing.assert_array_equal(
        respaldo, DetectorPlato({**config, "metodo": "umbral"}).detectar(imagen)
    )
    circulos = np.array([[[160, 160, 100], [60, 60, 40]]], dtype=np.float32)
    monkeypatch.setattr(cv2, "HoughCircles", lambda *args, **kwargs: circulos)
    mascara = DetectorPlato(config).detectar(imagen)
    assert mascara[160, 256] and not mascara[160, 259]
    assert not mascara[60, 60]


def test_umbral_componente_mayor_y_casco(datos_config: dict) -> None:
    """Descarta manchas externas y rellena los huecos de color mediante el casco."""
    imagen = np.full((320, 320, 3), 30, dtype=np.uint8)
    cv2.circle(imagen, (180, 180), 100, (230, 230, 230), -1)
    cv2.circle(imagen, (180, 180), 70, (0, 0, 220), -1)
    cv2.circle(imagen, (25, 25), 15, (230, 230, 230), -1)
    config = {**datos_config["plato"], "metodo": "umbral"}
    mascara = DetectorPlato(config).detectar(imagen)
    assert mascara[180, 180]
    assert not mascara[25, 25]


@pytest.mark.parametrize("metodo", ["hough", "umbral"])
def test_sin_plato_y_area_insuficiente(datos_config: dict, metodo: str) -> None:
    """Rechaza tanto una imagen oscura como un objeto demasiado pequeño."""
    config = {**datos_config["plato"], "metodo": metodo}
    imagen = np.zeros((320, 320, 3), dtype=np.uint8)
    with pytest.raises(PlatoNoDetectado, match="área suficiente"):
        DetectorPlato(config).detectar(imagen)
    cv2.circle(imagen, (160, 160), 20, (255, 255, 255), -1)
    with pytest.raises(PlatoNoDetectado):
        DetectorPlato(config).detectar(imagen)
