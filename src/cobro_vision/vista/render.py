"""Colores y superposición de regiones BGR sin dependencias de Tk."""

from hashlib import sha256

import cv2
import numpy as np

from cobro_vision.dominio.modelos import Region
from cobro_vision.dominio.utilidades import _validar_imagen, _validar_mascara


def paleta_clases(clases: list[str]) -> dict[str, tuple[int, int, int]]:
    """Asigna un color BGR estable, independiente del orden de las clases."""
    return {
        clase: tuple(64 + canal % 160 for canal in sha256(clase.encode()).digest()[:3])
        for clase in clases
    }


def dibujar_regiones(
    imagen_bgr: np.ndarray,
    regiones: list[Region],
    paleta: dict[str, tuple[int, int, int]],
) -> np.ndarray:
    """Devuelve una copia con relleno translúcido y contornos por clase."""
    _validar_imagen(imagen_bgr)
    salida = imagen_bgr.copy()
    for region in regiones:
        if region.estado == "eliminada" or region.resultado is None:
            continue
        color = paleta.get(region.resultado.id_clase)
        if color is None:
            continue
        _validar_mascara(region.mascara, imagen_bgr.shape[:2])
        capa = salida.copy()
        capa[region.mascara] = color
        salida = cv2.addWeighted(salida, 0.7, capa, 0.3, 0)
        contornos, _ = cv2.findContours(
            region.mascara.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(salida, contornos, -1, color, 1)
    return salida


def ajustar_a_caja(imagen: np.ndarray, ancho_max: int, alto_max: int) -> np.ndarray:
    """Reduce la imagen conservando la proporción, sin ampliar la entrada."""
    _validar_imagen(imagen)
    if ancho_max <= 0 or alto_max <= 0:
        raise ValueError("Las dimensiones de la caja deben ser positivas.")
    alto, ancho = imagen.shape[:2]
    escala = min(ancho_max / ancho, alto_max / alto, 1.0)
    dimensiones = (max(1, round(ancho * escala)), max(1, round(alto * escala)))
    return cv2.resize(imagen, dimensiones, interpolation=cv2.INTER_AREA)
