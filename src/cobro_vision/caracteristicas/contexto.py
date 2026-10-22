"""Conversiones de color y textura reutilizadas por imagen."""

from functools import cached_property

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from cobro_vision.dominio.utilidades import _validar_imagen


class ContextoImagen:
    """Conserva BGR y calcula cada representación una sola vez."""

    def __init__(self, bgr: np.ndarray) -> None:
        _validar_imagen(bgr)
        self.bgr = bgr
        self._cache_lbp: dict[tuple[int, int], np.ndarray] = {}

    @cached_property
    def hsv(self) -> np.ndarray:
        """Convierte a HSV uint8, con H entre 0 y 179."""
        return cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV)

    @cached_property
    def lab(self) -> np.ndarray:
        """Convierte a Lab uint8 con la codificación de OpenCV."""
        return cv2.cvtColor(self.bgr, cv2.COLOR_BGR2LAB)

    @cached_property
    def gris(self) -> np.ndarray:
        """Convierte a gris uint8."""
        return cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)

    def lbp(self, radio: int, puntos: int) -> np.ndarray:
        """Cachea LBP uniforme por pareja de radio y número de puntos."""
        if radio <= 0 or puntos <= 0:
            raise ValueError("El radio y los puntos de LBP deben ser positivos.")
        clave = (radio, puntos)
        if clave not in self._cache_lbp:
            self._cache_lbp[clave] = local_binary_pattern(
                self.gris, P=puntos, R=radio, method="uniform"
            )
        return self._cache_lbp[clave]

    @cached_property
    def _magnitud_gradiente(self) -> np.ndarray:
        horizontal = cv2.Sobel(self.gris, cv2.CV_64F, 1, 0)
        vertical = cv2.Sobel(self.gris, cv2.CV_64F, 0, 1)
        # hypot evita las aproximaciones SIMD de magnitude en float32,
        # que pueden variar con la alineación de memoria entre contextos.
        return np.hypot(horizontal, vertical)
