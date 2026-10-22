"""Contrato común de los descriptores de regiones."""

from abc import ABC, abstractmethod

import numpy as np

from cobro_vision.caracteristicas.contexto import ContextoImagen
from cobro_vision.dominio.modelos import Region


class ExtractorCaracteristicas(ABC):
    """Produce un vector numérico con un nombre estable por dimensión."""

    @abstractmethod
    def extraer(self, ctx: ContextoImagen, region: Region) -> np.ndarray:
        """Describe la región como un vector 1-D float32."""

    @abstractmethod
    def nombres(self) -> list[str]:
        """Devuelve un nombre por componente del vector."""
