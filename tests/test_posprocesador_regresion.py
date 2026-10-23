"""Referencia fija del plato sintético, capturada antes de la optimización."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cobro_vision.dominio.modelos import Region, ResultadoClasificacion
from cobro_vision.negocio.catalogo import CatalogoAlimentos
from cobro_vision.negocio.posprocesador import Posprocesador
from cobro_vision.segmentacion.base import _crear_region


def _predicciones_plato(mapa: np.ndarray) -> list[Region]:
    etiquetas = mapa.copy()
    etiquetas[etiquetas == 4] = 0
    etiquetas[60:90, 175:205] = 4
    etiquetas[110:160, 170:220] = 5
    etiquetas[180:183, 210:213] = 3
    etiquetas[180:183, 110:113] = 5
    etiquetas[140:144, 90:94] = 2
    etiquetas[30:33, 158:161] = 1
    clases = (
        "fondo",
        "pollo",
        "ensalada",
        "arroz_amarillo",
        "frijoles",
        "pescado_frito",
    )
    regiones = []
    # Celdas fijas: la referencia no depende de cambios de SLIC entre versiones.
    for y in range(0, mapa.shape[0], 32):
        for x in range(0, mapa.shape[1], 32):
            for indice, clase in enumerate(clases):
                mascara = np.zeros_like(mapa, dtype=bool)
                mascara[y : y + 32, x : x + 32] = (
                    etiquetas[y : y + 32, x : x + 32] == indice
                )
                if not mascara.any():
                    continue
                region = _crear_region(len(regiones) + 1, mascara)
                region.resultado = ResultadoClasificacion(
                    clase, (region.id % 11 + 1) / 10, True, "fondo"
                )
                regiones.append(region)
    return regiones


def _resumir(regiones: list[Region], posprocesador: Posprocesador) -> dict:
    return {
        "regiones": [
            {
                "id": region.id,
                "mascara_sha256": hashlib.sha256(region.mascara.tobytes()).hexdigest(),
                "bbox": list(region.bbox),
                "area": region.area,
                "vector": None if region.vector is None else region.vector.tolist(),
                "resultado": {
                    **asdict(region.resultado),
                    "score": region.resultado.score.hex(),
                },
                "estado": region.estado,
            }
            for region in regiones
        ],
        "eventos": posprocesador.eventos,
        "origenes": {str(k): v for k, v in posprocesador.origenes.items()},
    }


def test_plato_sintetico_coincide_exactamente_con_referencia_fija(
    plato_sintetico: tuple[np.ndarray, np.ndarray],
    catalogo: CatalogoAlimentos,
    ruta_proyecto: Path,
    datos_config: dict,
) -> None:
    """Conserva máscaras, scores exactos, orden, estados, eventos y procedencia."""
    _, mapa = plato_sintetico
    regiones = _predicciones_plato(mapa)
    originales = deepcopy(regiones)
    filas, columnas = np.indices(mapa.shape)
    plato = (filas - 160) ** 2 + (columnas - 160) ** 2 <= 140**2
    original_plato = plato.copy()
    posprocesador = Posprocesador(
        datos_config["posproceso"],
        catalogo,
        datos_config["clasificador"]["umbral_confianza"],
    )
    referencia = json.loads(
        (ruta_proyecto / "tests/fixtures/posprocesador_referencia.json").read_text(
            encoding="utf-8"
        )
    )
    for _ in range(2):
        salida = posprocesador.consolidar(regiones, plato)
        assert _resumir(salida, posprocesador) == referencia
    for region, original in zip(regiones, originales):
        np.testing.assert_array_equal(region.mascara, original.mascara)
        assert region.resultado == original.resultado
        assert region.estado == original.estado
        assert all(not np.shares_memory(region.mascara, r.mascara) for r in salida)
    np.testing.assert_array_equal(plato, original_plato)
