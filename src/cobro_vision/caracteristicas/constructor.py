"""Construcción uniforme de vectores para entrenamiento y operación."""

import numpy as np

from cobro_vision.caracteristicas.base import ExtractorCaracteristicas
from cobro_vision.caracteristicas.contexto import ContextoImagen
from cobro_vision.dominio.modelos import Region


class ConstructorVector:
    """Concatena los extractores recibidos sin elegir variantes concretas."""

    def __init__(self, extractores: list[ExtractorCaracteristicas]) -> None:
        self._extractores = list(extractores)

    def construir(self, ctx: ContextoImagen, region: Region) -> np.ndarray:
        """Construye y conserva el vector float32 de una región."""
        partes = [extractor.extraer(ctx, region) for extractor in self._extractores]
        vector = (
            np.concatenate(partes).astype(np.float32)
            if partes
            else np.empty(0, dtype=np.float32)
        )
        if vector.shape != (self.dimension(),) or not np.isfinite(vector).all():
            raise ValueError(
                "Los extractores devolvieron dimensión inválida o NaN/infinito."
            )
        region.vector = vector
        return vector

    def construir_lote(self, imagen: np.ndarray, regiones: list[Region]) -> np.ndarray:
        """Reutiliza un único contexto y devuelve una matriz (n, d)."""
        ctx = ContextoImagen(imagen)
        vectores = np.empty((len(regiones), self.dimension()), dtype=np.float32)
        for indice, region in enumerate(regiones):
            vectores[indice] = self.construir(ctx, region)
        return vectores

    def dimension(self) -> int:
        """Devuelve el número total de componentes del vector."""
        return len(self.nombres())

    def nombres(self) -> list[str]:
        """Concatena los nombres estables en el mismo orden que los vectores."""
        return [
            nombre for extractor in self._extractores for nombre in extractor.nombres()
        ]
