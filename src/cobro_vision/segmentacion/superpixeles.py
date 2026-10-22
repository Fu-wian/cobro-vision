"""Superpíxeles SLIC restringidos a la máscara del plato."""

from copy import deepcopy

import cv2
import numpy as np
from skimage.segmentation import slic

from cobro_vision.dominio.modelos import Region
from cobro_vision.dominio.utilidades import _validar_imagen, _validar_mascara
from cobro_vision.segmentacion.base import Segmentador, _crear_region


class SegmentadorSuperpixeles(Segmentador):
    """Construye regiones SLIC en orden de etiqueta, sin fusionar colores."""

    def __init__(self, config_segmentacion: dict) -> None:
        self._config = deepcopy(config_segmentacion)

    def segmentar(self, imagen: np.ndarray, mascara_plato: np.ndarray) -> list[Region]:
        """Convierte BGR a RGB para SLIC y descarta regiones pequeñas."""
        _validar_imagen(imagen)
        _validar_mascara(mascara_plato, imagen.shape[:2])
        cantidad = int(mascara_plato.sum())
        if cantidad < self._config["area_minima_px"]:
            return []
        # SLIC puede devolver etiqueta 0 para una máscara degenerada de un píxel.
        if cantidad == 1:
            return [_crear_region(1, mascara_plato.copy())]
        etiquetas = slic(
            cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB),
            n_segments=self._config["n_segmentos"],
            compactness=self._config["compacidad"],
            sigma=self._config["sigma"],
            start_label=1,
            mask=mascara_plato,
            enforce_connectivity=True,
        )
        regiones = []
        for etiqueta in np.unique(etiquetas):
            if etiqueta <= 0:
                continue
            mascara = (etiquetas == etiqueta) & mascara_plato
            if mascara.sum() >= self._config["area_minima_px"]:
                regiones.append(_crear_region(len(regiones) + 1, mascara))
        return regiones
