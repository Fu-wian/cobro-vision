"""Resúmenes independientes y explicaciones de las decisiones del pipeline."""

from pathlib import Path

import numpy as np
import pytest
from helpers import ClasificadorPorColorDominante

from cobro_vision.configuracion import Configuracion
from cobro_vision.controlador.fabrica import FabricaPipeline
from cobro_vision.dominio.modelos import Transaccion
from cobro_vision.explicacion.trazabilidad import Trazabilidad


def test_registrar_copia_datos_y_acepta_datos_opcionales() -> None:
    """Una entrada conserva el resumen y duración aunque cambie el diccionario original."""
    transaccion = Transaccion("T-prueba", np.zeros((2, 2, 3), dtype=np.uint8))
    trazabilidad = Trazabilidad()
    datos = {"ids": [1, 2]}
    trazabilidad.registrar(transaccion, "segmentacion", "Dos regiones.", datos, 12.5)
    datos["ids"].append(3)
    trazabilidad.registrar(transaccion, "cobro", "Recibo vacío.")
    assert transaccion.trazas[0].datos == {"ids": [1, 2]}
    assert transaccion.trazas[0].duracion_ms == 12.5
    assert (
        transaccion.trazas[1].datos == {} and transaccion.trazas[1].duracion_ms is None
    )


def test_explicaciones_incluyen_fuentes_area_umbral_y_politica(
    ruta_config_temporal: Path,
    plato_sintetico: tuple[np.ndarray, np.ndarray],
) -> None:
    """Las explicaciones usan los resultados y metadatos efectivos de la transacción."""
    controlador = FabricaPipeline.crear(Configuracion.cargar(ruta_config_temporal))
    controlador.clasificador = ClasificadorPorColorDominante(
        controlador.constructor.nombres()
    )
    transaccion = controlador.procesar(plato_sintetico[0])
    region = next(r for r in transaccion.regiones if r.resultado.id_clase == "pollo")
    texto = controlador.trazabilidad.explicar_how(transaccion, region.id)
    for esperado in (
        f"Región {region.id}",
        "% del plato",
        "SegmentadorSuperpixeles",
        str(controlador.constructor.dimension()),
        "pollo",
        "score",
        "0.5",
        "aceptada",
        "Ids de origen",
    ):
        assert esperado in texto
    origenes = controlador.posprocesador.origenes[region.id]
    assert len(origenes) > 1 and str(origenes) in texto
    texto = controlador.trazabilidad.explicar_why(transaccion, "pollo")
    for esperado in (
        "B/. 2.50",
        "CobroPorComponente",
        "Regiones de respaldo",
        "área total",
        f"{region.area} píxeles",
    ):
        assert esperado in texto
    with pytest.raises(ValueError, match="región"):
        controlador.trazabilidad.explicar_how(transaccion, 9999)
    with pytest.raises(ValueError, match="ítem"):
        controlador.trazabilidad.explicar_why(transaccion, "inexistente")
