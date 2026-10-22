"""Pruebas de datos, dinero exacto y mapas de máscaras sintéticas."""

from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from cobro_vision.dominio.errores import (
    ClaseDesconocida,
    CobroVisionError,
    ConfiguracionInvalida,
    EtiquetaInvalida,
    ModeloIncompatible,
    PlatoNoDetectado,
)
from cobro_vision.dominio.modelos import (
    Ajuste,
    Alimento,
    EntradaTraza,
    Etiquetado,
    Forma,
    Instancia,
    ItemRecibo,
    Muestra,
    Recibo,
    Region,
    Transaccion,
)
from cobro_vision.dominio.utilidades import formatear_dinero, mapa_desde_regiones


def test_dinero_exacto_y_alimento_inmutable() -> None:
    """Los importes siempre conservan centavos y el alimento es inmutable."""
    alimento = Alimento("pollo", "Pollo", "proteina", 250)
    assert alimento.precio == Decimal("2.50")
    with pytest.raises(FrozenInstanceError):
        alimento.precio_centavos = 99
    recibo = Recibo([ItemRecibo(alimento, "auto", [1], 0.5, True, 425)])
    assert type(recibo.total_centavos) is int
    assert recibo.total == Decimal("4.25")
    assert recibo.requiere_revision
    assert formatear_dinero(recibo.total_centavos) == "B/. 4.25"
    assert formatear_dinero(1) == "B/. 0.01"
    assert formatear_dinero(0, "USD") == "USD 0.00"
    assert Recibo([]).total_centavos == 0
    assert Recibo([]).total == Decimal(0)
    assert not Recibo([]).requiere_revision


@pytest.mark.parametrize("precio", [1.25, Decimal(100), True, -1, "100"])
def test_rechazar_precios_no_enteros(precio: object) -> None:
    """Ni float ni bool ni negativos pueden representar un precio."""
    with pytest.raises(ValueError):
        Alimento("pollo", "Pollo", "proteina", precio)
    with pytest.raises(ValueError):
        formatear_dinero(precio)


def test_mapa_de_regiones() -> None:
    """El mapa int32 conserva ids, ceros y el orden en solapes."""
    mascara = np.array([[True, True], [False, False]])
    otra = np.array([[False, True], [False, True]])
    regiones = [Region(2, mascara, (0, 0, 2, 1), 2), Region(7, otra, (1, 0, 2, 2), 2)]
    mapa = mapa_desde_regiones(regiones, (2, 2))
    np.testing.assert_array_equal(mapa, [[2, 7], [0, 7]])
    assert mapa.dtype == np.int32
    assert not mapa_desde_regiones([], (2, 3)).any()
    np.testing.assert_array_equal(mascara, [[True, True], [False, False]])


@pytest.mark.parametrize(
    "region",
    [
        Region(0, np.ones((2, 2), bool), (0, 0, 2, 2), 4),
        Region(1, np.ones((3, 3), bool), (0, 0, 3, 3), 9),
        Region(1, np.ones((2, 2), np.uint8), (0, 0, 2, 2), 4),
    ],
)
def test_mapa_rechaza_region_invalida(region: Region) -> None:
    """Las máscaras tienen forma y tipo compatibles e ids positivos."""
    with pytest.raises(ValueError):
        mapa_desde_regiones([region], (2, 2))


def test_listas_independientes_y_tipos_dataset() -> None:
    """Las transacciones no comparten listas y los datos no procesan imágenes."""
    imagen = np.zeros((2, 2, 3), dtype=np.uint8)
    primera = Transaccion("T-1", imagen)
    segunda = Transaccion("T-2", imagen)
    primera.ajustes.append(Ajuste("agregar", "pollo"))
    primera.trazas.append(EntradaTraza("cobro", "Prueba", {}))
    primera.tiempos["cobro"] = 1.0
    assert segunda.ajustes == segunda.trazas == segunda.regiones == []
    assert segunda.tiempos == {}
    forma = Forma("pollo", "polygon", [(0, 0), (1, 0), (1, 1)])
    etiqueta = Etiquetado(Path("foto.png"), Path("foto.json"), 2, 2, [forma])
    muestra = Muestra(
        "S01/P01_a", "P01", "S01", "train", etiqueta.ruta_imagen, etiqueta
    )
    instancia = Instancia(
        np.zeros(3, np.float32), "pollo", muestra.plato_id, muestra.id_foto, "poligono"
    )
    assert muestra.etiquetado.formas == [forma]
    assert instancia.plato_id == "P01"


@pytest.mark.parametrize(
    "error",
    [
        ClaseDesconocida,
        ConfiguracionInvalida,
        EtiquetaInvalida,
        ModeloIncompatible,
        PlatoNoDetectado,
    ],
)
def test_jerarquia_errores(error: type[Exception]) -> None:
    """Todos los errores propios admiten captura común."""
    assert isinstance(error("prueba"), CobroVisionError)
