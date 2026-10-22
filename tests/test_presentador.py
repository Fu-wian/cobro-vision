"""Presentación pura del cajero, sin importar Tk ni crear una ventana."""

import subprocess
import sys

import pytest

from cobro_vision.controlador.controlador import ControladorTransaccion
from cobro_vision.dominio.modelos import Alimento, Recibo, Transaccion
from cobro_vision.vista.presentador import (
    clases_visibles,
    etiquetas_alimentos,
    filas_recibo,
    mensajes_alerta,
    opciones_correccion,
    puede_confirmar,
    puede_editar,
    texto_explicacion,
    texto_tiempos,
    texto_total,
)


def test_importar_presentador_y_render_no_importa_tk() -> None:
    """Un proceso sin ventana puede importar ambos módulos sin cargar tkinter."""
    resultado = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import cobro_vision.vista.presentador; "
                "import cobro_vision.vista.render; "
                "assert 'tkinter' not in sys.modules; "
                "assert 'customtkinter' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert resultado.returncode == 0, resultado.stderr


def test_filas_y_total_exactos_sin_mutar_recibo(
    transaccion_cajero: Transaccion,
) -> None:
    """Las filas muestran los precios cobrados y la revisión del recibo original."""
    t = transaccion_cajero
    filas = filas_recibo(t.recibo, "B/.")
    assert [f.id_clase for f in filas] == ["arroz_amarillo", "ensalada", "pollo"]
    assert [f.precio_texto for f in filas] == ["B/. 1.00", "B/. 0.75", "B/. 2.50"]
    assert filas[-1].estado == "revisar" and filas[-1].origen == "auto"
    assert all(f.estado == "ok" for f in filas[:2])
    assert texto_total(t.recibo, "B/.") == "B/. 4.25"
    assert texto_total(None, "$ ".strip()) == "$ 0.00"
    assert filas_recibo(Recibo([]), "B/.") == []
    t.recibo.items[-1].precio_cobrado_centavos = 0
    assert filas_recibo(t.recibo, "B/.")[-1].precio_texto == "B/. 0.00"


@pytest.mark.parametrize(
    ("estado", "revision", "exigir", "esperado"),
    [
        ("requiere_revision", True, True, False),
        ("requiere_revision", True, False, True),
        ("procesada", False, True, True),
        ("procesada", True, True, False),
        ("confirmada", False, True, False),
        ("error", False, False, False),
        ("nueva", False, False, False),
    ],
)
def test_habilitar_confirmacion(
    transaccion_cajero: Transaccion,
    estado: str,
    revision: bool,
    exigir: bool,
    esperado: bool,
) -> None:
    """El estado y la política de revisión deciden si se puede confirmar."""
    t = transaccion_cajero
    t.estado = estado
    for item in t.recibo.items:
        item.requiere_revision = revision
    assert puede_confirmar(t, exigir) is esperado
    t.recibo = Recibo([])
    assert not puede_confirmar(t, exigir)
    t.recibo = None
    assert not puede_confirmar(t, exigir)
    assert not puede_editar(None)


def test_alternativa_primero_y_excluir_fondo_inactivas_y_desconocidas(
    controlador_cajero: ControladorTransaccion, transaccion_cajero: Transaccion
) -> None:
    """Las opciones no duplican alternativas y permiten aceptar la clase actual."""
    catalogo = controlador_cajero.catalogo
    t = transaccion_cajero
    catalogo.agregar(Alimento("inactivo", "Inactivo", "base", 100, activo=False))
    opciones = opciones_correccion(catalogo, t, "pollo")
    assert opciones[0] == "pescado_frito"
    assert (
        "pollo" in opciones and "fondo" not in opciones and "inactivo" not in opciones
    )
    assert len(opciones) == len(set(opciones))
    t.regiones[0].resultado.alternativa = "fondo"
    t.regiones[1].resultado.alternativa = "inexistente"
    assert opciones_correccion(catalogo, t, "pollo") == list(
        etiquetas_alimentos(catalogo)
    )
    controlador_cajero.agregar_item(t, "frijoles")
    assert opciones_correccion(catalogo, t, "frijoles") == list(
        etiquetas_alimentos(catalogo)
    )
    assert t.regiones[0].resultado.id_clase == "pollo"


def test_mensajes_explicaciones_leyenda_y_tiempos(
    controlador_cajero: ControladorTransaccion, transaccion_cajero: Transaccion
) -> None:
    """Los textos de alerta, trazabilidad y tiempos se construyen fuera de Tk."""
    t = transaccion_cajero
    assert "Revise" in mensajes_alerta(t)[0]
    assert "total: 12.25 ms" in texto_tiempos(t)
    assert clases_visibles(t) == ["arroz_amarillo", "ensalada", "pollo"]
    explicacion = texto_explicacion(t, controlador_cajero.trazabilidad)
    assert "Región 1" in explicacion and "Región 4" in explicacion
    assert "B/. 2.50" in explicacion and "Regiones de respaldo: [1, 2]" in explicacion
    controlador_cajero.corregir_item(t, "pollo", "pollo")
    assert mensajes_alerta(t) == []
    t.estado = "confirmada"
    assert "guardada" in mensajes_alerta(t)[0]
    t.estado = "error"
    t.mensaje_error = "No se detectó el plato"
    assert mensajes_alerta(t) == ["No se detectó el plato"]
    t.mensaje_error = None
    assert mensajes_alerta(t) == ["No se pudo procesar la foto."]
    t.estado = "nueva"
    t.recibo = None
    assert "foto" in mensajes_alerta(t)[0]
    t.recibo = Recibo([])
    assert "No hay alimentos" in mensajes_alerta(t)[0]


def test_leyenda_excluye_regiones_eliminadas_y_pendientes(
    transaccion_cajero: Transaccion,
) -> None:
    """La leyenda representa solamente clases que tienen regiones visibles."""
    t = transaccion_cajero
    t.regiones[0].estado = "eliminada"
    t.regiones[1].resultado = None
    assert clases_visibles(t) == ["arroz_amarillo", "ensalada"]
