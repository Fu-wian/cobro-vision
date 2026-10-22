"""Exportación de los artefactos de depuración de la CLI."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from cobro_vision.dominio.modelos import Region, Transaccion


def guardar_depuracion(
    transaccion: Transaccion, carpeta: Path, texto_recibo: str
) -> None:
    """Guarda cinco imágenes, el recibo y una traza JSON sin matrices."""
    carpeta.mkdir(parents=True, exist_ok=True)
    imagen = transaccion.imagen_procesada
    if imagen is None:
        imagen = transaccion.imagen_original
    mascara = transaccion.mascara_plato
    if mascara is None:
        mascara = np.zeros(imagen.shape[:2], dtype=bool)
    imagenes = {
        "01_procesada.png": imagen,
        "02_mascara_plato.png": mascara.astype(np.uint8) * 255,
        "03_regiones_crudas.png": _dibujar(imagen, transaccion.regiones_crudas, False),
        "04_clasificadas.png": _dibujar(imagen, transaccion.regiones_crudas, True),
        "05_consolidadas.png": _dibujar(imagen, transaccion.regiones, True),
    }
    for nombre, salida in imagenes.items():
        if not cv2.imwrite(str(carpeta / nombre), salida):
            raise OSError(
                f"No se pudo escribir la imagen de depuración: {carpeta / nombre}"
            )
    (carpeta / "recibo.txt").write_text(texto_recibo + "\n", encoding="utf-8")
    datos = {
        "id": transaccion.id,
        "estado": transaccion.estado,
        "mensaje_error": transaccion.mensaje_error,
        "tiempos": transaccion.tiempos,
        "trazas": [asdict(entrada) for entrada in transaccion.trazas],
    }
    (carpeta / "traza.json").write_text(
        json.dumps(datos, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _dibujar(imagen: np.ndarray, regiones: list[Region], por_clase: bool) -> np.ndarray:
    salida = imagen.copy()
    for region in regiones:
        if por_clase and region.resultado is not None:
            resumen_hash = hashlib.sha256(
                region.resultado.id_clase.encode("utf-8")
            ).digest()
            color = tuple(64 + int(c) % 192 for c in resumen_hash[:3])
            salida[region.mascara] = np.rint(
                0.55 * imagen[region.mascara] + 0.45 * np.asarray(color)
            ).astype(np.uint8)
        else:
            color = (255, 255, 255)
        contornos, _ = cv2.findContours(
            region.mascara.astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(salida, contornos, -1, color, 1)
    return salida
