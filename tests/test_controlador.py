"""Pipeline completo sin entrenamiento y manejo del plato ausente."""

import re
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
import yaml
from helpers import ClasificadorPorColorDominante, generar_plato_sintetico

from cobro_vision.adquisicion.camara import CamaraArchivo
from cobro_vision.configuracion import Configuracion
from cobro_vision.controlador.controlador import ControladorTransaccion
from cobro_vision.controlador.fabrica import FabricaPipeline

ETAPAS = {
    "preprocesamiento",
    "deteccion_plato",
    "segmentacion",
    "caracteristicas",
    "clasificacion",
    "posprocesamiento",
    "cobro",
}


@pytest.fixture
def controlador(ruta_config_temporal: Path) -> ControladorTransaccion:
    """Usa dependencias reales y sustituye únicamente el clasificador de prueba."""
    controlador = FabricaPipeline.crear(Configuracion.cargar(ruta_config_temporal))
    controlador.clasificador = ClasificadorPorColorDominante(
        controlador.constructor.nombres()
    )
    return controlador


@pytest.mark.parametrize(("sectores", "centavos"), [(3, 425), (4, 500)])
def test_pipeline_sintetico_total_exacto_siete_etapas_y_reproducibilidad(
    controlador: ControladorTransaccion,
    sectores: int,
    centavos: int,
    ruta_proyecto: Path,
) -> None:
    """Una foto atraviesa visión y cobro con clases correctas y precios enteros."""
    imagen = (
        CamaraArchivo.leer(ruta_proyecto / "tests/fixtures/plato_sintetico.png")
        if sectores == 4
        else generar_plato_sintetico(controlador.config.semilla, sectores)[0]
    )
    original = imagen.copy()
    transaccion = controlador.procesar(imagen)
    assert transaccion.estado == "procesada"
    assert re.fullmatch(r"T-\d{8}-\d{6}-[0-9a-f]{4}", transaccion.id)
    assert transaccion.recibo.total_centavos == centavos
    assert transaccion.recibo.total == Decimal(centavos) / 100
    esperadas = {"pollo", "ensalada", "arroz_amarillo"}
    if sectores == 4:
        esperadas.add("frijoles")
    assert {i.alimento.id_clase for i in transaccion.recibo.items} == esperadas
    assert len(transaccion.recibo.items) == sectores
    assert set(transaccion.tiempos) == ETAPAS | {"total"}
    assert all(t >= 0 for t in transaccion.tiempos.values())
    assert transaccion.tiempos["total"] == pytest.approx(
        sum(transaccion.tiempos[e] for e in ETAPAS)
    )
    etapas_traza = [e for e in transaccion.trazas if e.duracion_ms is not None]
    assert [e.etapa for e in etapas_traza] == [
        "preprocesamiento",
        "deteccion_plato",
        "segmentacion",
        "caracteristicas",
        "clasificacion",
        "posprocesamiento",
        "cobro",
    ]
    assert all(e.duracion_ms == transaccion.tiempos[e.etapa] for e in etapas_traza)
    for item in transaccion.recibo.items:
        area = sum(r.area for r in transaccion.regiones if r.id in item.ids_region)
        assert item.area_frac == pytest.approx(area / transaccion.mascara_plato.sum())
    assert len(transaccion.regiones_crudas) > len(transaccion.regiones)
    assert all(
        r.vector is not None and r.resultado is not None
        for r in transaccion.regiones_crudas
    )
    assert all(r.estado == "pendiente" for r in transaccion.regiones_crudas)
    assert all(
        cruda is not final and not np.shares_memory(cruda.mascara, final.mascara)
        for cruda in transaccion.regiones_crudas
        for final in transaccion.regiones
    )
    repetida = controlador.procesar(imagen)
    assert repetida.id != transaccion.id
    assert repetida.recibo == transaccion.recibo
    assert len(repetida.regiones) == len(transaccion.regiones)
    for region, otra in zip(transaccion.regiones, repetida.regiones):
        np.testing.assert_array_equal(region.mascara, otra.mascara)
        assert region.resultado == otra.resultado
    np.testing.assert_array_equal(imagen, original)


def test_plato_ausente_devuelve_error_legible_recibo_vacio_y_tiempos_propios(
    controlador: ControladorTransaccion,
    plato_sintetico: tuple[np.ndarray, np.ndarray],
) -> None:
    """El fallo esperado no propaga excepción ni conserva datos de la foto anterior."""
    controlador.procesar(plato_sintetico[0])
    transaccion = controlador.procesar(np.zeros((320, 320, 3), dtype=np.uint8))
    assert transaccion.estado == "error" and transaccion.mensaje_error
    assert transaccion.recibo.items == [] and transaccion.recibo.total_centavos == 0
    assert transaccion.regiones == [] and transaccion.regiones_crudas == []
    assert transaccion.mascara_plato is None
    assert set(transaccion.tiempos) == {"preprocesamiento", "deteccion_plato", "total"}
    assert transaccion.trazas[-1].datos["error"] == transaccion.mensaje_error


def test_estado_revision_y_pipeline_sin_regiones(
    ruta_config_temporal: Path, plato_sintetico: tuple[np.ndarray, np.ndarray]
) -> None:
    """El estado sigue al recibo y un segmentador sin regiones produce cero."""
    datos = yaml.safe_load(ruta_config_temporal.read_text(encoding="utf-8"))
    datos["clasificador"]["umbral_confianza"] = 11
    ruta_config_temporal.write_text(yaml.safe_dump(datos), encoding="utf-8")
    controlador = FabricaPipeline.crear(Configuracion.cargar(ruta_config_temporal))
    controlador.clasificador = ClasificadorPorColorDominante(
        controlador.constructor.nombres()
    )
    transaccion = controlador.procesar(plato_sintetico[0])
    assert (
        transaccion.estado == "requiere_revision"
        and transaccion.recibo.requiere_revision
    )
    datos["segmentacion"]["area_minima_px"] = 320 * 320 + 1
    ruta_config_temporal.write_text(yaml.safe_dump(datos), encoding="utf-8")
    vacia = FabricaPipeline.crear(Configuracion.cargar(ruta_config_temporal)).procesar(
        plato_sintetico[0]
    )
    assert vacia.estado == "procesada" and vacia.recibo.total_centavos == 0
    assert set(vacia.tiempos) == ETAPAS | {"total"}


def test_capturar_y_procesar_usa_camara_configurada(
    ruta_config_temporal: Path, ruta_proyecto: Path
) -> None:
    """La captura de archivo pasa por el mismo pipeline sin seleccionar otra cámara."""
    datos = yaml.safe_load(ruta_config_temporal.read_text(encoding="utf-8"))
    datos["camara"]["carpeta"] = str(ruta_proyecto / "tests/fixtures")
    ruta_config_temporal.write_text(yaml.safe_dump(datos), encoding="utf-8")
    controlador = FabricaPipeline.crear(Configuracion.cargar(ruta_config_temporal))
    transaccion = controlador.capturar_y_procesar()
    assert transaccion.estado in {"procesada", "requiere_revision"}
    assert transaccion.imagen_original.shape == (320, 320, 3)
