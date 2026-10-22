"""Contrato común para clasificar vectores de características."""

from abc import ABC, abstractmethod

import numpy as np

from cobro_vision.dominio.modelos import ResultadoClasificacion


class Clasificador(ABC):
    """Predice clases de una región o un lote de regiones."""

    @abstractmethod
    def predecir_lote(self, vectores: np.ndarray) -> list[ResultadoClasificacion]:
        """Devuelve una predicción por fila de la matriz (n, d)."""

    def predecir(self, vector: np.ndarray) -> ResultadoClasificacion:
        """Delega la predicción individual al mismo procedimiento por lotes."""
        if vector.ndim != 1:
            raise ValueError("Se esperaba un vector de una dimensión.")
        return self.predecir_lote(vector.reshape(1, -1))[0]
