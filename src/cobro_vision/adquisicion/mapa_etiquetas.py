"""Rasterización ordenada de etiquetas a la escala de procesamiento."""

import math

import cv2
import numpy as np

from cobro_vision.dominio.errores import EtiquetaInvalida
from cobro_vision.dominio.modelos import Etiquetado


class MapaEtiquetas:
    """Asocia las clases con índices int16 desde 1; reserva 0 para lo no etiquetado."""

    def __init__(self, clases: list[str]) -> None:
        if (
            len(clases) != len(set(clases))
            or len(clases) > np.iinfo(np.int16).max
            or any(not isinstance(clase, str) or not clase.strip() for clase in clases)
        ):
            raise ValueError("Las clases deben ser únicas, no vacías y caber en int16.")
        self._clases = list(clases)
        self._indices = {clase: indice for indice, clase in enumerate(clases, start=1)}

    def rasterizar(
        self, etiquetado: Etiquetado, shape_destino: tuple[int, int], escala: float
    ) -> np.ndarray:
        """Multiplica los puntos originales por escala; en los solapes gana la última."""
        if not math.isfinite(escala) or not 0 < escala <= 1:
            raise ValueError("escala debe estar entre 0 (excluido) y 1.")
        esperado = (
            max(1, round(etiquetado.alto * escala)),
            max(1, round(etiquetado.ancho * escala)),
        )
        if shape_destino != esperado:
            raise ValueError(
                f"shape_destino {shape_destino} no coincide con escala: {esperado}."
            )
        mapa = np.zeros(shape_destino, dtype=np.int16)
        for numero, forma in enumerate(etiquetado.formas, start=1):
            if forma.etiqueta not in self._indices:
                raise EtiquetaInvalida(
                    f"Forma {numero}: etiqueta desconocida {forma.etiqueta!r}."
                )
            if forma.tipo not in ("polygon", "rectangle") or len(forma.puntos) < 3:
                raise EtiquetaInvalida(
                    f"Forma {numero}: se requiere un polígono válido."
                )
            originales = np.asarray(forma.puntos, dtype=np.float64)
            if (
                originales.shape != (len(forma.puntos), 2)
                or not np.isfinite(originales).all()
            ):
                raise EtiquetaInvalida(f"Forma {numero}: coordenadas no válidas.")
            if (
                (originales < 0).any()
                or (originales[:, 0] >= etiquetado.ancho).any()
                or (originales[:, 1] >= etiquetado.alto).any()
            ):
                raise EtiquetaInvalida(
                    f"Forma {numero}: puntos fuera de la imagen original."
                )
            # TODO(equipo): se redondea al píxel más cercano y se recorta el borde
            # tras reducir, igual que las dimensiones redondeadas del preprocesador.
            puntos = np.rint(originales * escala).astype(np.int32)
            puntos[:, 0] = np.clip(puntos[:, 0], 0, shape_destino[1] - 1)
            puntos[:, 1] = np.clip(puntos[:, 1], 0, shape_destino[0] - 1)
            cv2.fillPoly(mapa, [puntos], self._indices[forma.etiqueta])
        return mapa

    def clase_de(self, indice: int) -> str | None:
        """Devuelve None para 0 o un índice que no pertenece al mapa."""
        return self._clases[indice - 1] if 1 <= indice <= len(self._clases) else None
