"""Correcciones cronológicas, estado de revisión y persistencia de confirmación."""

import csv
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from cobro_vision.configuracion import Configuracion
from cobro_vision.controlador.controlador import ControladorTransaccion
from cobro_vision.dominio.errores import ClaseDesconocida
from cobro_vision.dominio.modelos import Ajuste, Transaccion


def test_corregir_region_conserva_otras_regiones_y_revision(
    controlador_cajero: ControladorTransaccion, transaccion_cajero: Transaccion
) -> None:
    """Una sola región se corrige y la otra conserva su revisión pendiente."""
    t = transaccion_cajero
    crudas = deepcopy(t.regiones)
    t.regiones_crudas = crudas
    assert controlador_cajero.corregir_region(t, 1, "pescado_frito") is t
    assert t.recibo.total_centavos == 700
    assert t.estado == "requiere_revision"
    assert t.regiones[0].estado == "corregida"
    assert not t.regiones[0].resultado.requiere_revision
    assert t.regiones[1].estado == "revisar"
    assert t.ajustes == [Ajuste("corregir", "pescado_frito", 1)]
    assert t.regiones_crudas[0].resultado.id_clase == "pollo"
    assert t.recibo.items[-1].area_frac == pytest.approx(0.25)
    controlador_cajero.corregir_region(t, 2, "pescado_frito")
    assert t.recibo.total_centavos == 450 and t.estado == "procesada"
    assert not t.recibo.requiere_revision
    assert t.ajustes[-1] == Ajuste("corregir", "pescado_frito", 2)


def test_corregir_item_varias_regiones_recalcula_una_vez(
    controlador_cajero: ControladorTransaccion, transaccion_cajero: Transaccion
) -> None:
    """Un ítem con dos regiones cambia completo con un solo recálculo."""
    t = transaccion_cajero
    with patch.object(
        controlador_cajero.calculadora,
        "calcular",
        wraps=controlador_cajero.calculadora.calcular,
    ) as calcular:
        controlador_cajero.corregir_item(t, "pollo", "pescado_frito")
    calcular.assert_called_once_with(t.regiones, t.ajustes)
    assert t.recibo.total_centavos == 450
    assert t.estado == "procesada"
    pescado = next(i for i in t.recibo.items if i.alimento.id_clase == "pescado_frito")
    assert pescado.ids_region == [1, 2] and pescado.area_frac == pytest.approx(0.5)
    assert t.ajustes == [
        Ajuste("corregir", "pescado_frito", 1),
        Ajuste("corregir", "pescado_frito", 2),
    ]


def test_aceptar_misma_clase_y_fusionar_con_item_existente(
    controlador_cajero: ControladorTransaccion, transaccion_cajero: Transaccion
) -> None:
    """Elegir la misma clase acepta la revisión; corregir a otra no duplica cobro."""
    t = transaccion_cajero
    controlador_cajero.corregir_item(t, "pollo", "pollo")
    assert t.recibo.total_centavos == 425 and t.estado == "procesada"
    controlador_cajero.corregir_item(t, "arroz_amarillo", "ensalada")
    assert t.recibo.total_centavos == 325
    assert len(t.recibo.items) == 2


def test_quitar_agregar_manual_corregir_y_sin_duplicados(
    controlador_cajero: ControladorTransaccion, transaccion_cajero: Transaccion
) -> None:
    """La baja elimina todas las regiones y el alta posterior tiene origen manual."""
    t = transaccion_cajero
    controlador_cajero.quitar_item(t, "pollo")
    assert t.recibo.total_centavos == 175 and t.estado == "procesada"
    assert all(r.estado == "eliminada" for r in t.regiones[:2])
    controlador_cajero.agregar_item(t, "pollo")
    controlador_cajero.agregar_item(t, "pollo")
    assert t.recibo.total_centavos == 425
    item = next(i for i in t.recibo.items if i.alimento.id_clase == "pollo")
    assert item.origen == "manual" and item.ids_region == [] and item.area_frac is None
    with patch.object(
        controlador_cajero.calculadora,
        "calcular",
        wraps=controlador_cajero.calculadora.calcular,
    ) as calcular:
        controlador_cajero.corregir_item(t, "pollo", "pescado_frito")
    assert calcular.call_count == 1
    assert t.recibo.total_centavos == 450
    assert t.ajustes[-2:] == [
        Ajuste("quitar", "pollo"),
        Ajuste("agregar", "pescado_frito"),
    ]
    assert t.recibo.items[-1].origen == "manual"


def test_corregir_a_clase_previamente_quitada_recupera_regiones_actuales(
    controlador_cajero: ControladorTransaccion, transaccion_cajero: Transaccion
) -> None:
    """Una baja histórica no oculta una corrección posterior hacia esa clase."""
    t = transaccion_cajero
    controlador_cajero.quitar_item(t, "ensalada")
    controlador_cajero.agregar_item(t, "ensalada")
    controlador_cajero.corregir_item(t, "pollo", "ensalada")
    ensalada = next(i for i in t.recibo.items if i.alimento.id_clase == "ensalada")
    assert t.recibo.total_centavos == 175
    assert ensalada.ids_region == [1, 2] and ensalada.origen == "auto"
    assert t.regiones[3].estado == "eliminada"
    controlador_cajero.quitar_item(t, "ensalada")
    controlador_cajero.agregar_item(t, "frijoles")
    assert t.recibo.total_centavos == 175
    assert {i.alimento.id_clase for i in t.recibo.items} == {
        "arroz_amarillo",
        "frijoles",
    }


def test_altas_historicas_no_reaparecen_despues_de_corregir_un_item_auto(
    controlador_cajero: ControladorTransaccion, transaccion_cajero: Transaccion
) -> None:
    """Altas duplicadas y manuales fusionadas siguen al ítem al cambiar de clase."""
    t = transaccion_cajero
    controlador_cajero.agregar_item(t, "pollo")
    controlador_cajero.corregir_item(t, "pollo", "pescado_frito")
    assert t.recibo.total_centavos == 450
    assert "pollo" not in {i.alimento.id_clase for i in t.recibo.items}
    controlador_cajero.agregar_item(t, "frijoles")
    controlador_cajero.corregir_item(t, "frijoles", "pescado_frito")
    controlador_cajero.corregir_item(t, "pescado_frito", "ensalada")
    assert t.recibo.total_centavos == 175
    assert {i.alimento.id_clase for i in t.recibo.items} == {
        "arroz_amarillo",
        "ensalada",
    }


def test_alta_manual_fusionada_no_reaparece_al_mover_ultima_region(
    controlador_cajero: ControladorTransaccion, transaccion_cajero: Transaccion
) -> None:
    """La corrección individual conserva el alta mientras queden regiones de la clase."""
    t = transaccion_cajero
    controlador_cajero.agregar_item(t, "frijoles")
    controlador_cajero.corregir_item(t, "pollo", "frijoles")
    controlador_cajero.corregir_region(t, 1, "ensalada")
    assert {i.alimento.id_clase for i in t.recibo.items} == {
        "arroz_amarillo",
        "ensalada",
        "frijoles",
    }
    controlador_cajero.corregir_region(t, 2, "ensalada")
    assert {i.alimento.id_clase for i in t.recibo.items} == {
        "arroz_amarillo",
        "ensalada",
    }


@pytest.mark.parametrize(
    "operacion",
    [
        lambda c, t: c.corregir_region(t, 100, "pollo"),
        lambda c, t: c.corregir_region(t, 1, "inexistente"),
        lambda c, t: c.corregir_item(t, "pollo", "inexistente"),
        lambda c, t: c.corregir_item(t, "inexistente", "pollo"),
        lambda c, t: c.corregir_item(t, "pollo", "fondo"),
        lambda c, t: c.quitar_item(t, "inexistente"),
        lambda c, t: c.agregar_item(t, "inexistente"),
        lambda c, t: c.agregar_item(t, "fondo"),
    ],
)
def test_ajustes_invalidos_no_modifican_transaccion(
    controlador_cajero: ControladorTransaccion,
    transaccion_cajero: Transaccion,
    operacion,
) -> None:
    """La validación ocurre antes de modificar regiones, recibo o ajustes."""
    t = transaccion_cajero
    with pytest.raises((ValueError, ClaseDesconocida)):
        operacion(controlador_cajero, t)
    assert t.ajustes == [] and t.estado == "requiere_revision"
    assert t.recibo.total_centavos == 425
    assert t.regiones[0].resultado.id_clase == "pollo"


def test_corregir_region_a_fondo_y_rechazar_region_eliminada(
    controlador_cajero: ControladorTransaccion, transaccion_cajero: Transaccion
) -> None:
    """Se puede reclasificar una región como fondo; una eliminada no es editable."""
    t = transaccion_cajero
    controlador_cajero.corregir_region(t, 3, "fondo")
    assert t.recibo.total_centavos == 325
    controlador_cajero.quitar_item(t, "pollo")
    with pytest.raises(ValueError, match="activa"):
        controlador_cajero.corregir_region(t, 1, "pollo")


def test_confirmacion_csv_exacta_idempotente_y_bloqueo_posterior(
    controlador_cajero: ControladorTransaccion, transaccion_cajero: Transaccion
) -> None:
    """El CSV guarda centavos y JSON legible, con una sola fila por confirmación."""
    t = transaccion_cajero
    ruta = (
        Path(controlador_cajero.config.seccion("rutas")["salidas"])
        / "transacciones.csv"
    )
    with pytest.raises(ValueError, match="Revise"):
        controlador_cajero.confirmar(t)
    assert not ruta.exists()
    controlador_cajero.corregir_item(t, "pollo", "pollo")
    assert controlador_cajero.confirmar(t) is t and t.estado == "confirmada"
    controlador_cajero.confirmar(t)
    otra = deepcopy(t)
    otra.id = "T-20261021-120001-abce"
    otra.estado = "procesada"
    controlador_cajero.confirmar(otra)
    with ruta.open(encoding="utf-8", newline="") as archivo:
        filas = list(csv.DictReader(archivo))
    assert len(filas) == 2 and filas[0]["id"] == t.id
    assert filas[0]["total_centavos"] == "425" and filas[0]["n_ajustes"] == "2"
    assert json.loads(filas[0]["tiempos"]) == t.tiempos
    assert "T" in filas[0]["fecha_hora"]
    items = json.loads(filas[0]["items"])
    assert sum(i["precio_cobrado_centavos"] for i in items) == 425
    assert items[-1]["ids_region"] == [1, 2]
    for accion in [
        lambda: controlador_cajero.agregar_item(t, "pollo"),
        lambda: controlador_cajero.quitar_item(t, "pollo"),
        lambda: controlador_cajero.corregir_region(t, 1, "pollo"),
        lambda: controlador_cajero.corregir_item(t, "pollo", "ensalada"),
    ]:
        with pytest.raises(ValueError):
            accion()


def test_confirmar_con_revision_opcional_y_fallo_de_escritura(
    controlador_cajero: ControladorTransaccion, transaccion_cajero: Transaccion
) -> None:
    """La política de revisión se respeta y un fallo de disco no confirma."""
    t = transaccion_cajero
    controlador_cajero.config = Configuracion(
        {
            **controlador_cajero.config._datos,
            "ui": {**controlador_cajero.config.seccion("ui"), "exigir_revision": False},
        }
    )
    with (
        patch.object(Path, "open", side_effect=OSError("Disco lleno")),
        pytest.raises(OSError, match="Disco lleno"),
    ):
        controlador_cajero.confirmar(t)
    assert t.estado == "requiere_revision"
    controlador_cajero.confirmar(t)
    assert t.estado == "confirmada"


@pytest.mark.parametrize("estado", ["nueva", "error"])
def test_no_editar_o_confirmar_transaccion_no_procesada(
    controlador_cajero: ControladorTransaccion,
    transaccion_cajero: Transaccion,
    estado: str,
) -> None:
    """Los estados que no tienen un resultado válido no admiten operaciones."""
    transaccion_cajero.estado = estado
    with pytest.raises(ValueError):
        controlador_cajero.agregar_item(transaccion_cajero, "pollo")
    with pytest.raises(ValueError):
        controlador_cajero.confirmar(transaccion_cajero)


def test_recibo_vacio_no_se_confirma(
    controlador_cajero: ControladorTransaccion, transaccion_cajero: Transaccion
) -> None:
    """Un recibo vacío puede corregirse pero no registrarse como cobro."""
    t = transaccion_cajero
    for clase in ["pollo", "arroz_amarillo", "ensalada"]:
        controlador_cajero.quitar_item(t, clase)
    with pytest.raises(ValueError, match="al menos"):
        controlador_cajero.confirmar(t)


def test_cargar_y_procesar_delega_lectura_y_pipeline(
    controlador_cajero: ControladorTransaccion, ruta_proyecto: Path
) -> None:
    """La ruta elegida usa el mismo controlador de la captura normal."""
    t = controlador_cajero.cargar_y_procesar(
        ruta_proyecto / "tests/fixtures/plato_sintetico.png"
    )
    assert t.estado in {"procesada", "requiere_revision"}
    assert t.imagen_original.dtype == np.uint8
