"""Clasificador determinista de demostración, sin entrenamiento."""

import numpy as np

from cobro_vision.clasificacion.base import Clasificador
from cobro_vision.dominio.modelos import ResultadoClasificacion


class ClasificadorFalso(Clasificador):
    """Asigna clases ficticias usando la semilla y el tono medio del vector."""

    def __init__(self, clases: list[str], semilla: int = 0) -> None:
        if not clases or len(set(clases)) != len(clases):
            raise ValueError("Se requieren clases distintas y una lista no vacía.")
        self._clases = list(clases)
        self._semilla = semilla
        # TODO(equipo): el contrato no recibe nombres; la fábrica los proporciona
        # por este atributo. Sin ellos se interpreta el vector como valores de H;
        # sin características HSV se usa su media como sustituto ficticio de H.
        self.nombres_caracteristicas: list[str] = []

    def predecir_lote(self, vectores: np.ndarray) -> list[ResultadoClasificacion]:
        """Devuelve resultados estables incluso al dividir o reordenar el lote."""
        if vectores.ndim != 2 or not np.isfinite(vectores).all():
            raise ValueError("Se esperaba una matriz (n, d) finita.")
        nombres = self.nombres_caracteristicas
        indices_h = [
            i
            for i, nombre in enumerate(nombres)
            if nombre.startswith("hsv_h_") and nombre.removeprefix("hsv_h_").isdigit()
        ]
        if nombres and len(nombres) != vectores.shape[1]:
            raise ValueError("Los nombres no coinciden con la dimensión del vector.")
        if "hsv_h_media" in nombres:
            medias = vectores[:, nombres.index("hsv_h_media")] * 180
        elif indices_h:
            histogramas = vectores[:, indices_h]
            centros = (np.arange(len(indices_h)) + 0.5) * 180 / len(indices_h)
            masas = histogramas.sum(axis=1)
            medias = np.divide(
                histogramas @ centros,
                masas,
                out=np.zeros(len(vectores)),
                where=masas != 0,
            )
        else:
            medias = (
                vectores.mean(axis=1) if vectores.shape[1] else np.zeros(len(vectores))
            )
        resultados = []
        for media in medias:
            indice = (self._semilla + int(np.floor(float(media)))) % len(self._clases)
            resultados.append(
                ResultadoClasificacion(
                    self._clases[indice],
                    10.0,
                    False,
                    self._clases[(indice + 1) % len(self._clases)]
                    if len(self._clases) > 1
                    else None,
                )
            )
        return resultados
