"""Descriptores clásicos LBP, gradiente de Sobel y GLCM."""

from copy import deepcopy

import numpy as np
from skimage.feature import graycomatrix, graycoprops

from cobro_vision.caracteristicas.base import ExtractorCaracteristicas
from cobro_vision.caracteristicas.contexto import ContextoImagen
from cobro_vision.dominio.modelos import Region
from cobro_vision.dominio.utilidades import _validar_mascara


class ExtractorTextura(ExtractorCaracteristicas):
    """Combina LBP uniforme con medidas opcionales de gradiente y GLCM."""

    def __init__(self, config: dict) -> None:
        self._config = deepcopy(config)

    def extraer(self, ctx: ContextoImagen, region: Region) -> np.ndarray:
        """Describe solo la región y mantiene dimensión fija aunque esté vacía."""
        _validar_mascara(region.mascara, ctx.bgr.shape[:2])
        if not region.mascara.any():
            return np.zeros(len(self.nombres()), dtype=np.float32)
        lbp = self._config["lbp"]
        bins = lbp["puntos"] + 2
        codigos = ctx.lbp(lbp["radio"], lbp["puntos"])[region.mascara]
        histograma, _ = np.histogram(codigos, bins=bins, range=(0, bins))
        partes = list(histograma / histograma.sum())
        if self._config["gradiente"]:
            magnitud = ctx._magnitud_gradiente[region.mascara]
            partes.extend(
                (magnitud.mean(), magnitud.std(), ctx.gris[region.mascara].std())
            )
        if self._config["glcm"]:
            partes.extend(self._extraer_glcm(ctx, region))
        return np.asarray(partes, dtype=np.float32)

    def _extraer_glcm(self, ctx: ContextoImagen, region: Region) -> list[float]:
        x0, y0, x1, y1 = region.bbox
        recorte = ctx.gris[y0:y1, x0:x1]
        mascara = region.mascara[y0:y1, x0:x1]
        relleno = ctx.gris[region.mascara].mean()
        cuantizado = (np.where(mascara, recorte, relleno) / 16).astype(np.uint8)
        # TODO(equipo): GLCM usa distancia 1 y ángulo 0 por defecto, simétrica;
        # si se configuran varios valores, cada propiedad se promedia entre ellos.
        matriz = graycomatrix(
            cuantizado,
            distances=self._config.get("glcm_distancias", [1]),
            angles=self._config.get("glcm_angulos", [0.0]),
            levels=16,
            symmetric=True,
            normed=True,
        )
        return [
            float(graycoprops(matriz, propiedad).mean())
            for propiedad in ("contrast", "homogeneity", "energy", "correlation")
        ]

    def nombres(self) -> list[str]:
        """Conserva nombres y orden para todas las opciones de textura."""
        nombres = [
            f"lbp_{indice:02d}" for indice in range(self._config["lbp"]["puntos"] + 2)
        ]
        if self._config["gradiente"]:
            nombres.extend(("sobel_media", "sobel_desviacion", "gris_desviacion"))
        if self._config["glcm"]:
            nombres.extend(
                (
                    "glcm_contraste",
                    "glcm_homogeneidad",
                    "glcm_energia",
                    "glcm_correlacion",
                )
            )
        return nombres
