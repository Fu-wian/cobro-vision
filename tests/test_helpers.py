"""Verifica el origen artificial y reproducible de la imagen versionada."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from helpers import escribir_dataset_sintetico, generar_plato_sintetico

from cobro_vision.adquisicion.lector_labelme import LectorLabelMe
from cobro_vision.adquisicion.mapa_etiquetas import MapaEtiquetas


def test_fixture_generada_por_codigo(
    ruta_proyecto: Path,
    datos_config: dict,
    plato_sintetico: tuple[np.ndarray, np.ndarray],
) -> None:
    """El PNG coincide píxel a píxel con el generador y la semilla del proyecto."""
    imagen, mapa = plato_sintetico
    guardada = cv2.imread(str(ruta_proyecto / "tests/fixtures/plato_sintetico.png"))
    np.testing.assert_array_equal(guardada, imagen)
    repetida, mapa_repetido = generar_plato_sintetico(datos_config["semilla"], 4)
    np.testing.assert_array_equal(repetida, imagen)
    np.testing.assert_array_equal(mapa, mapa_repetido)
    assert mapa.dtype == np.int16
    assert set(np.unique(mapa)) == {0, 1, 2, 3, 4}
    assert imagen[mapa == 1, 2].mean() > imagen[mapa == 1, 0].mean() + 100
    assert imagen[mapa == 2, 1].mean() > imagen[mapa == 2, 2].mean() + 100
    assert imagen[mapa == 3, 1].mean() > 200 and imagen[mapa == 3, 2].mean() > 200
    assert imagen[mapa == 4, 2].mean() > imagen[mapa == 4, 0].mean()
    assert imagen[0, 0].mean() == pytest.approx(85, abs=8)
    assert imagen[25, 160].mean() > 230


@pytest.mark.parametrize("sectores", [0, 1, 2, 3, 4])
def test_cantidad_sectores(datos_config: dict, sectores: int) -> None:
    """Cada etiqueta positiva corresponde a un sector solicitado."""
    imagen, mapa = generar_plato_sintetico(datos_config["semilla"], sectores)
    assert imagen.dtype == np.uint8
    assert set(np.unique(mapa)) == set(range(sectores + 1))


@pytest.mark.parametrize("sectores", [-1, 5, 2.5, True])
def test_sectores_invalidos(datos_config: dict, sectores: object) -> None:
    """El generador rechaza cantidades ambiguas o fuera de los cuatro colores."""
    with pytest.raises(ValueError, match="sectores"):
        generar_plato_sintetico(datos_config["semilla"], sectores)


def test_dataset_sintetico_coherente_y_reproducible(tmp_path: Path) -> None:
    """Todos los alimentos se anotan con contornos acordes con su imagen artificial."""
    carpeta = tmp_path / "raw"
    escribir_dataset_sintetico(carpeta, 2)
    archivos = sorted(ruta for ruta in carpeta.rglob("*") if ruta.is_file())
    originales = {ruta: ruta.read_bytes() for ruta in archivos}
    assert len(archivos) == 8
    mapa = MapaEtiquetas(["pollo", "ensalada", "arroz_amarillo", "frijoles", "fondo"])
    for ruta in carpeta.rglob("*.json"):
        etiquetado = LectorLabelMe().leer(ruta)
        imagen = cv2.imread(str(etiquetado.ruta_imagen))
        etiquetas = mapa.rasterizar(etiquetado, imagen.shape[:2], 1)
        assert set(np.unique(etiquetas)) == {0, 1, 2, 3, 4, 5}
        assert imagen[etiquetas == 1, 2].mean() > imagen[etiquetas == 1, 0].mean() + 100
        assert imagen[etiquetas == 2, 1].mean() > imagen[etiquetas == 2, 2].mean() + 100
    escribir_dataset_sintetico(carpeta, 2)
    assert {ruta: ruta.read_bytes() for ruta in archivos} == originales
