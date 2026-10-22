"""Histogramas y momentos de color sobre los píxeles de una región."""

from copy import deepcopy

import numpy as np

from cobro_vision.caracteristicas.base import ExtractorCaracteristicas
from cobro_vision.caracteristicas.contexto import ContextoImagen
from cobro_vision.dominio.modelos import Region
from cobro_vision.dominio.utilidades import _validar_mascara


class ExtractorHistogramaColor(ExtractorCaracteristicas):
    """Concatena histogramas normalizados de HSV o Lab y sus momentos."""

    def __init__(self, config: dict) -> None:
        self._config = deepcopy(config)
        if self._config["espacio"] not in ("hsv", "lab"):
            raise ValueError("El espacio de color debe ser hsv o lab.")

    def extraer(self, ctx: ContextoImagen, region: Region) -> np.ndarray:
        """Usa solo la máscara y devuelve ceros cuando está vacía."""
        _validar_mascara(region.mascara, ctx.bgr.shape[:2])
        if not region.mascara.any():
            return np.zeros(len(self.nombres()), dtype=np.float32)
        espacio = self._config["espacio"]
        pixeles = getattr(ctx, espacio)[region.mascara]
        rangos = (180, 256, 256) if espacio == "hsv" else (256, 256, 256)
        partes = []
        for canal, (bins, rango) in enumerate(zip(self._config["bins"], rangos)):
            histograma, _ = np.histogram(pixeles[:, canal], bins=bins, range=(0, rango))
            partes.extend(histograma / histograma.sum())
        if self._config["momentos"]:
            # TODO(equipo): los momentos usan el límite superior del rango
            # del histograma (180/256), ya que no se precisa si usar 179/255.
            for canal, rango in enumerate(rangos):
                partes.extend(
                    (pixeles[:, canal].mean() / rango, pixeles[:, canal].std() / rango)
                )
        return np.asarray(partes, dtype=np.float32)

    def nombres(self) -> list[str]:
        """Nombra histogramas y momentos en el orden de extracción."""
        espacio = self._config["espacio"]
        canales = ("h", "s", "v") if espacio == "hsv" else ("l", "a", "b")
        nombres = [
            f"{espacio}_{canal}_{indice:02d}"
            for canal, bins in zip(canales, self._config["bins"])
            for indice in range(bins)
        ]
        if self._config["momentos"]:
            nombres.extend(
                f"{espacio}_{canal}_{momento}"
                for canal in canales
                for momento in ("media", "desviacion")
            )
        return nombres
