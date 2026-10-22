"""Pruebas de agrupación, ajustes de recibo y políticas de precios."""

from decimal import Decimal

import numpy as np
import pytest

from cobro_vision.dominio.errores import ClaseDesconocida
from cobro_vision.dominio.modelos import (
    Ajuste,
    ItemRecibo,
    Region,
    ResultadoClasificacion,
)
from cobro_vision.negocio.calculadora import CalculadoraCosto
from cobro_vision.negocio.catalogo import CatalogoAlimentos
from cobro_vision.negocio.politicas import CobroPorComponente, CobroPorPlato


def crear_region(
    indice: int, clase: str, revision: bool = False, estado: str = "aceptada"
) -> Region:
    """Genera una máscara sintética de un píxel en una fila de ocho."""
    mascara = np.zeros((1, 8), dtype=bool)
    mascara[0, indice - 1] = True
    return Region(
        indice,
        mascara,
        (indice - 1, 0, indice, 1),
        1,
        resultado=ResultadoClasificacion(clase, 1.0, revision),
        estado=estado,
    )


def test_total_semilla_y_un_item_por_clase(catalogo: CatalogoAlimentos) -> None:
    """Dos regiones de pollo cobran una vez: 250 + 100 + 75 = 425."""
    regiones = [
        crear_region(1, "pollo"),
        crear_region(2, "pollo", True),
        crear_region(3, "arroz_amarillo"),
        crear_region(4, "ensalada"),
        crear_region(5, "fondo"),
        crear_region(6, "pescado_frito", estado="eliminada"),
    ]
    recibo = CalculadoraCosto(catalogo, CobroPorComponente()).calcular(regiones)
    assert len(recibo.items) == 3
    assert recibo.total_centavos == 425
    assert recibo.total == Decimal("4.25")
    pollo = next(item for item in recibo.items if item.alimento.id_clase == "pollo")
    assert pollo.ids_region == [1, 2]
    assert pollo.area_frac == pytest.approx(2 / 6)
    assert pollo.requiere_revision and recibo.requiere_revision
    assert pollo.origen == "auto"
    assert [item.alimento.id_clase for item in recibo.items] == sorted(
        item.alimento.id_clase for item in recibo.items
    )


def test_ajustes_agregar_quitar_sin_duplicados(catalogo: CatalogoAlimentos) -> None:
    """Las altas manuales no duplican clases y quitar permite volver a agregar."""
    calculadora = CalculadoraCosto(catalogo, CobroPorComponente())
    regiones = [crear_region(1, "pollo", True)]
    ajustes = [
        Ajuste("agregar", "pollo"),
        Ajuste("quitar", "pollo"),
        Ajuste("agregar", "ensalada"),
        Ajuste("agregar", "ensalada"),
        Ajuste("agregar", "fondo"),
    ]
    recibo = calculadora.calcular(regiones, ajustes)
    assert recibo.total_centavos == 75 and len(recibo.items) == 1
    assert recibo.items[0].origen == "manual"
    assert recibo.items[0].ids_region == [] and recibo.items[0].area_frac is None
    assert not recibo.requiere_revision
    assert regiones[0].resultado.id_clase == "pollo"
    assert calculadora.calcular(regiones).total_centavos == 250


def test_sin_regiones_clasificadas_y_clase_desconocida(
    catalogo: CatalogoAlimentos,
) -> None:
    """Un recibo vacío cuesta cero y una predicción desconocida es un error."""
    calculadora = CalculadoraCosto(catalogo, CobroPorComponente())
    assert calculadora.calcular([]).total_centavos == 0
    pendiente = crear_region(1, "pollo")
    pendiente.resultado = None
    assert calculadora.calcular([pendiente]).items == []
    with pytest.raises(ClaseDesconocida):
        calculadora.calcular([crear_region(1, "inexistente")])
    with pytest.raises(ClaseDesconocida):
        calculadora.calcular([], [Ajuste("agregar", "inexistente")])
    with pytest.raises(ClaseDesconocida):
        calculadora.calcular([], [Ajuste("quitar", "inexistente")])


def test_por_plato_tarifa_y_conservacion_de_origen(catalogo: CatalogoAlimentos) -> None:
    """La proteína cobra el plato y el resto conserva sus datos con precio cero."""
    regiones = [crear_region(1, "pollo", True), crear_region(2, "arroz_amarillo")]
    recibo = CalculadoraCosto(catalogo, CobroPorPlato({"pollo": 450})).calcular(
        regiones
    )
    assert recibo.total_centavos == 450
    assert recibo.requiere_revision
    precios = {
        item.alimento.id_clase: item.precio_cobrado_centavos for item in recibo.items
    }
    assert precios == {"pollo": 450, "arroz_amarillo": 0}
    assert next(
        item for item in recibo.items if item.alimento.id_clase == "pollo"
    ).ids_region == [1]


@pytest.mark.parametrize("precios", [{}, {"pescado_frito": 475}])
def test_por_plato_sin_tarifa_usa_suma(
    catalogo: CatalogoAlimentos, precios: dict[str, int]
) -> None:
    """Sin proteína con tarifa se cobran los componentes habituales."""
    regiones = [
        crear_region(1, "pollo"),
        crear_region(2, "arroz_amarillo"),
        crear_region(3, "ensalada"),
    ]
    assert (
        CalculadoraCosto(catalogo, CobroPorPlato(precios))
        .calcular(regiones)
        .total_centavos
        == 425
    )


def test_por_plato_varias_proteinas_y_tarifa_cero(catalogo: CatalogoAlimentos) -> None:
    """Varias proteínas usan una tarifa determinista; cero también es tarifa."""
    regiones = [crear_region(1, "pollo"), crear_region(2, "pescado_frito")]
    calculadora = CalculadoraCosto(
        catalogo, CobroPorPlato({"pollo": 450, "pescado_frito": 475})
    )
    assert calculadora.calcular(regiones).total_centavos == 475
    assert calculadora.calcular(list(reversed(regiones))).total_centavos == 475
    assert (
        CalculadoraCosto(catalogo, CobroPorPlato({"pollo": 0}))
        .calcular(regiones)
        .total_centavos
        == 0
    )


def test_politicas_sin_mutar_items_y_sin_fondo(catalogo: CatalogoAlimentos) -> None:
    """Aplicar políticas no cambia los ítems recibidos ni permite cobrar fondo."""
    items = [
        ItemRecibo(catalogo.obtener("pollo"), "auto", [1], 0.5, False, 0),
        ItemRecibo(catalogo.obtener("fondo"), "auto", [2], 0.5, False, 999),
    ]
    assert CobroPorComponente().calcular(items).total_centavos == 250
    assert CobroPorPlato({"pollo": 450}).calcular(items).total_centavos == 450
    assert items[0].precio_cobrado_centavos == 0


@pytest.mark.parametrize("precio", [450.0, True, -1])
def test_tarifa_por_plato_exige_centavos(precio: object) -> None:
    """Una tarifa de plato tiene las mismas restricciones monetarias."""
    with pytest.raises(ValueError):
        CobroPorPlato({"pollo": precio})
