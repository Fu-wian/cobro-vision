"""Generación sintética, máscaras a escala, pureza y muestreo reproducible."""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from cobro_vision.adquisicion.camara import CamaraArchivo
from cobro_vision.adquisicion.gestor_dataset import GestorDataset
from cobro_vision.configuracion import Configuracion
from cobro_vision.controlador.fabrica import FabricaPipeline
from cobro_vision.dominio.errores import PlatoNoDetectado
from cobro_vision.negocio.catalogo import CatalogoAlimentos
from cobro_vision.segmentacion.base import _crear_region


def test_modos_y_parches_deterministas(
    config_entrenamiento: Configuracion,
    catalogo: CatalogoAlimentos,
) -> None:
    """Ambos concatena los dos orígenes y la foto determina sus parches."""
    muestras = GestorDataset(config_entrenamiento, catalogo).cargar_muestras("train")[
        :2
    ]
    constructor = FabricaPipeline.crear_entrenador(
        config_entrenamiento
    ).constructor_instancias
    poligonos = constructor.generar(muestras, "poligonos")
    regiones = constructor.generar(muestras, "segmentacion")
    ambos = constructor.generar(muestras, "ambos")
    repetidas = constructor.generar(list(reversed(muestras)), "poligonos")
    assert {i.origen for i in poligonos} == {"poligono", "parche"}
    assert {i.origen for i in regiones} == {"region"}
    assert len(ambos) == len(poligonos) + len(regiones)
    for primero, segundo in zip(poligonos, repetidas, strict=True):
        np.testing.assert_array_equal(primero.vector, segundo.vector)
        assert (primero.id_foto, primero.plato_id, primero.id_clase) == (
            segundo.id_foto,
            segundo.plato_id,
            segundo.id_clase,
        )
        assert primero.vector.dtype == np.float32 and np.isfinite(primero.vector).all()
    for muestra in muestras:
        esperadas = [i for i in poligonos + regiones if i.id_foto == muestra.id_foto]
        obtenidas = [i for i in ambos if i.id_foto == muestra.id_foto]
        for esperada, obtenida in zip(esperadas, obtenidas, strict=True):
            assert esperada.origen == obtenida.origen
            np.testing.assert_array_equal(esperada.vector, obtenida.vector)
    with pytest.raises(ValueError, match="modo"):
        constructor.generar(muestras, "aumentadas")


def test_poligonos_a_escala_y_mismos_vectores_que_operacion(
    config_entrenamiento: Configuracion,
    catalogo: CatalogoAlimentos,
) -> None:
    """El mapa se reduce con el mismo factor y las máscaras alimentan el constructor común."""
    datos = config_entrenamiento._datos
    datos["imagen"]["lado_mayor"] = 160
    datos["entrenamiento"]["parches_por_poligono"] = 0
    config = Configuracion(datos)
    constructor = FabricaPipeline.crear_entrenador(config).constructor_instancias
    muestra = GestorDataset(config, catalogo).cargar_muestras("train")[0]
    original = CamaraArchivo.leer(muestra.ruta_imagen)
    controlador = FabricaPipeline.crear(config)
    procesada = controlador.preprocesador.procesar(original)
    assert procesada.shape == (160, 160, 3)
    mapa = constructor.mapa.rasterizar(muestra.etiquetado, (160, 160), 0.5)
    instancias = constructor.generar([muestra], "poligonos")
    esperados = []
    for indice in np.unique(mapa):
        cantidad, componentes = cv2.connectedComponents(
            (mapa == indice).astype(np.uint8)
        )
        for componente in range(1, cantidad):
            mascara = componentes == componente
            if mascara.sum() >= config.seccion("segmentacion")["area_minima_px"]:
                esperados.append(_crear_region(len(esperados) + 1, mascara))
    vectores = controlador.constructor.construir_lote(procesada, esperados)
    np.testing.assert_array_equal([i.vector for i in instancias], vectores)
    assert constructor.constructor.nombres() == controlador.constructor.nombres()


def test_pureza_no_ignora_pixeles_sin_etiqueta(
    config_entrenamiento: Configuracion,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Una región con 50% etiquetado se rechaza aunque esa mitad sea de una sola clase."""
    constructor = FabricaPipeline.crear_entrenador(
        config_entrenamiento
    ).constructor_instancias
    mapa = np.zeros((20, 20), dtype=np.int16)
    mapa[:, :10] = 1
    region = _crear_region(1, np.ones(mapa.shape, dtype=bool))
    with (
        patch.object(constructor.segmentador, "segmentar", return_value=[region]),
        caplog.at_level("INFO"),
    ):
        assert (
            constructor._regiones(
                np.zeros((20, 20, 3), dtype=np.uint8), region.mascara, mapa, "foto"
            )
            == []
        )
    assert "1 descartadas" in caplog.text
    mapa[:, 10:16] = 1  # Exactamente el 80% requerido.
    with patch.object(constructor.segmentador, "segmentar", return_value=[region]):
        assert (
            len(
                constructor._regiones(
                    np.zeros((20, 20, 3), dtype=np.uint8), region.mascara, mapa, "foto"
                )
            )
            == 1
        )


def test_fallo_deteccion_se_advierte_sin_perder_poligonos(
    config_entrenamiento: Configuracion,
    catalogo: CatalogoAlimentos,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Una detección fallida se omite en segmentación; ambos conserva el otro modo."""
    constructor = FabricaPipeline.crear_entrenador(
        config_entrenamiento
    ).constructor_instancias
    muestras = GestorDataset(config_entrenamiento, catalogo).cargar_muestras("train")[
        :1
    ]
    with patch.object(
        constructor.detector, "detectar", side_effect=PlatoNoDetectado("sin plato")
    ):
        assert constructor.generar(muestras, "segmentacion") == []
        assert constructor.generar(muestras, "ambos")
    assert muestras[0].id_foto in caplog.text and "sin plato" in caplog.text


def test_parches_pequenos_y_componentes_se_descartan(
    config_entrenamiento: Configuracion,
) -> None:
    """No se usan componentes bajo el mínimo ni discos con menos de la mitad de área."""
    constructor = FabricaPipeline.crear_entrenador(
        config_entrenamiento
    ).constructor_instancias
    mapa = np.zeros((100, 100), dtype=np.int16)
    mapa[1:4, 1:4] = 1  # Menor que area_minima_px.
    mapa[30:50, 30:50] = 1  # 400 píxeles < 50% del disco de radio 22.
    filas = constructor._poligonos(mapa, "foto")
    assert len(filas) == 1 and filas[0][2] == "poligono"
    assert filas[0][0].area == 400
