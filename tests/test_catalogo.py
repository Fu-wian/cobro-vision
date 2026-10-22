"""Pruebas de persistencia, semilla y centavos enteros."""

import sqlite3
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from cobro_vision.dominio.errores import ClaseDesconocida
from cobro_vision.dominio.modelos import Alimento
from cobro_vision.negocio.catalogo import CatalogoAlimentos


def test_inicializar_idempotente_y_upsert(
    catalogo: CatalogoAlimentos, ruta_semilla_catalogo: Path
) -> None:
    """Repetir la inicialización actualiza sin duplicar filas."""
    catalogo.actualizar_precio("pollo", 999)
    catalogo.inicializar(ruta_semilla_catalogo)
    catalogo.inicializar(ruta_semilla_catalogo)
    assert catalogo.clases_validas() == [
        "arroz_amarillo",
        "ensalada",
        "fondo",
        "frijoles",
        "pescado_frito",
        "pollo",
    ]
    with sqlite3.connect(catalogo.ruta_db) as conexion:
        assert conexion.execute("SELECT COUNT(*) FROM alimentos").fetchone()[0] == 6
        assert (
            conexion.execute(
                "SELECT typeof(precio_centavos) FROM alimentos WHERE id_clase='pollo'"
            ).fetchone()[0]
            == "integer"
        )
    pollo = catalogo.obtener("pollo")
    assert type(pollo.precio_centavos) is int
    assert pollo.precio_centavos == 250
    assert pollo.precio == Decimal("2.50")
    assert catalogo.obtener("fondo").precio_centavos == 0


def test_clases_activas_y_cobrables(catalogo: CatalogoAlimentos) -> None:
    """Las consultas excluyen clases inactivas y separan fondo de cobrables."""
    catalogo.agregar(replace(catalogo.obtener("pollo"), activo=False))
    for nombre in ("pollo", "desconocida", "' OR 1=1 --"):
        with pytest.raises(
            ClaseDesconocida, match=nombre if nombre == "pollo" else None
        ):
            catalogo.obtener(nombre)
    clases = catalogo.clases_validas()
    assert clases == sorted(clases)
    assert "fondo" in clases and "pollo" not in clases
    assert "fondo" not in [
        alimento.id_clase for alimento in catalogo.listar(solo_cobrables=True)
    ]


def test_agregar_y_actualizar(catalogo: CatalogoAlimentos) -> None:
    """Las altas y los cambios de precio persisten en otra instancia."""
    catalogo.agregar(Alimento("prueba", "Ejemplo sintético", "base", 123))
    catalogo.actualizar_precio("prueba", 456)
    assert CatalogoAlimentos(catalogo.ruta_db).obtener("prueba").precio_centavos == 456
    with pytest.raises(ClaseDesconocida):
        catalogo.actualizar_precio("ausente", 100)


@pytest.mark.parametrize("precio", [1.0, True, -1, "100"])
def test_actualizar_rechaza_precio_invalido(
    catalogo: CatalogoAlimentos, precio: object
) -> None:
    """SQLite no recibe precios float, negativos ni booleanos."""
    with pytest.raises(ValueError):
        catalogo.actualizar_precio("pollo", precio)
    assert catalogo.obtener("pollo").precio_centavos == 250


@pytest.mark.parametrize(
    "fila",
    [
        "pollo,Pollo,proteina,2.50,1",
        "pollo,Pollo,proteina,-1,1",
        "pollo,Pollo,proteina,250,si",
        "pollo,Pollo,otra,250,1",
        "pollo,Pollo,proteina,250,1,extra",
    ],
)
def test_semilla_invalida_no_modifica_catalogo(
    catalogo: CatalogoAlimentos, tmp_path: Path, fila: str
) -> None:
    """Una fila inválida impide aplicar también las filas anteriores."""
    ruta = tmp_path / "invalida.csv"
    ruta.write_text(
        "id_clase,nombre,categoria,precio_centavos,activo\nensalada,Ensalada,ensalada,1,1\n"
        + fila
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fila 3"):
        catalogo.inicializar(ruta)
    assert catalogo.obtener("ensalada").precio_centavos == 75
