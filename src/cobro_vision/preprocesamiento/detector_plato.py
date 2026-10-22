"""Detección geométrica del plato sin modelos entrenados."""

from copy import deepcopy

import cv2
import numpy as np

from cobro_vision.dominio.errores import PlatoNoDetectado
from cobro_vision.dominio.utilidades import _validar_imagen


class DetectorPlato:
    """Obtiene una máscara por Hough, umbral o imagen completa."""

    def __init__(self, config_plato: dict) -> None:
        self._config = deepcopy(config_plato)

    def detectar(self, imagen: np.ndarray) -> np.ndarray:
        """Devuelve una máscara bool o informa que no hay plato suficiente."""
        _validar_imagen(imagen)
        metodo = self._config["metodo"]
        if metodo == "imagen_completa":
            mascara = np.ones(imagen.shape[:2], dtype=bool)
        elif metodo == "hough":
            mascara = self._hough(imagen)
            if mascara is None:
                mascara = self._umbral(imagen)
        elif metodo == "umbral":
            mascara = self._umbral(imagen)
        else:
            raise ValueError(f"Método de detección desconocido: {metodo}")
        if (
            not mascara.any()
            or mascara.sum() < self._config["area_minima_frac"] * mascara.size
        ):
            raise PlatoNoDetectado("No se detectó un plato con área suficiente.")
        return mascara

    def _hough(self, imagen: np.ndarray) -> np.ndarray | None:
        gris = cv2.medianBlur(cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY), 5)
        lado = min(gris.shape)
        circulos = cv2.HoughCircles(
            gris,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=lado / 2,
            param1=120,
            param2=50,
            minRadius=int(self._config["radio_min_frac"] * lado),
            maxRadius=int(self._config["radio_max_frac"] * lado),
        )
        if circulos is None:
            return None
        x, y, radio = circulos[0, 0]
        mascara = np.zeros(gris.shape, dtype=np.uint8)
        cv2.circle(
            mascara,
            (round(float(x)), round(float(y))),
            round(float(radio) * self._config["reducir_radio"]),
            1,
            -1,
        )
        return mascara.astype(bool)

    def _umbral(self, imagen: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        umbral = self._config["umbral"]
        mascara = (
            (hsv[:, :, 1] < umbral["s_max"]) & (hsv[:, :, 2] > umbral["v_min"])
        ).astype(np.uint8)
        for operacion, tamano in ((cv2.MORPH_CLOSE, 15), (cv2.MORPH_OPEN, 5)):
            nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamano, tamano))
            mascara = cv2.morphologyEx(mascara, operacion, nucleo)
        cantidad, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(mascara)
        if cantidad <= 1:
            return np.zeros(mascara.shape, dtype=bool)
        mayor = 1 + int(np.argmax(estadisticas[1:, cv2.CC_STAT_AREA]))
        filas, columnas = np.nonzero(etiquetas == mayor)
        puntos = np.column_stack((columnas, filas)).astype(np.int32)
        casco = cv2.convexHull(puntos)
        salida = np.zeros(mascara.shape, dtype=np.uint8)
        cv2.fillConvexPoly(salida, casco, 1)
        return salida.astype(bool)
