"""Contrato de segmentación y construcción común de regiones."""

from abc import ABC, abstractmethod

import numpy as np

from cobro_vision.dominio.modelos import Region


class Segmentador(ABC):
    """Divide el área del plato en regiones para describir sus píxeles."""

    @abstractmethod
    def segmentar(self, imagen: np.ndarray, mascara_plato: np.ndarray) -> list[Region]:
        """Devuelve regiones con máscaras del tamaño de la imagen."""


def _crear_region(identificador: int, mascara: np.ndarray) -> Region:
    filas, columnas = np.nonzero(mascara)
    return Region(
        id=identificador,
        mascara=mascara,
        bbox=(
            int(columnas.min()),
            int(filas.min()),
            int(columnas.max()) + 1,
            int(filas.max()) + 1,
        ),
        area=int(mascara.sum()),
    )
