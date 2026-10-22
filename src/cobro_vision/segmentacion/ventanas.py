"""Segmentación determinista mediante celdas de una cuadrícula."""

from copy import deepcopy

import numpy as np

from cobro_vision.dominio.modelos import Region
from cobro_vision.dominio.utilidades import _validar_imagen, _validar_mascara
from cobro_vision.segmentacion.base import Segmentador, _crear_region


class SegmentadorVentanas(Segmentador):
    """Recorre ventanas por filas y columnas dentro del plato."""

    def __init__(self, config_segmentacion: dict) -> None:
        self._config = deepcopy(config_segmentacion)

    def segmentar(self, imagen: np.ndarray, mascara_plato: np.ndarray) -> list[Region]:
        """Devuelve regiones disjuntas, recortadas y con área suficiente."""
        _validar_imagen(imagen)
        _validar_mascara(mascara_plato, imagen.shape[:2])
        alto, ancho = mascara_plato.shape
        tamano = self._config["ventanas"]["tamano"]
        paso = self._config["ventanas"]["paso"]
        ocupada = np.zeros_like(mascara_plato)
        regiones = []
        for y in range(0, alto, paso):
            for x in range(0, ancho, paso):
                mascara = np.zeros_like(mascara_plato)
                mascara[y : y + tamano, x : x + tamano] = mascara_plato[
                    y : y + tamano, x : x + tamano
                ]
                # TODO(equipo): si paso < tamano, el solape pertenece a la primera
                # celda aceptada para mantener las regiones disjuntas (sección 11).
                mascara &= ~ocupada
                if mascara.sum() >= self._config["area_minima_px"]:
                    regiones.append(_crear_region(len(regiones) + 1, mascara))
                    ocupada |= mascara
        return regiones
