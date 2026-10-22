"""Humo gráfico y aislamiento del hilo de trabajo, también sin pantalla."""

import gc
from functools import partial
from queue import Queue
from threading import Event, get_ident
from time import monotonic
from tkinter import TclError
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from cobro_vision.controlador.controlador import ControladorTransaccion
from cobro_vision.dominio.modelos import Transaccion
from cobro_vision.vista.vista_cajero import VistaCajero


@pytest.fixture
def vista(controlador_cajero: ControladorTransaccion):
    """Crea una ventana real y omite solo la ausencia de pantalla."""
    try:
        ventana = VistaCajero(controlador_cajero, controlador_cajero.config)
    except TclError as error:
        mensaje = str(error).lower()
        if "no display name" in mensaje or "couldn't connect to display" in mensaje:
            pytest.skip(f"Sin pantalla disponible: {error}")
        raise
    ventana.withdraw()
    try:
        yield ventana
    finally:
        ventana.destroy()


def _esperar_resultado(vista: VistaCajero) -> None:
    """Atiende el bucle real de Tk mientras espera el resultado del trabajador."""
    limite = monotonic() + 5
    consulta = None

    def revisar() -> None:
        nonlocal consulta
        consulta = None
        if not vista._ocupada or monotonic() >= limite:
            vista.quit()
        else:
            consulta = vista.after(5, revisar)

    consulta = vista.after(0, revisar)
    try:
        vista.mainloop()
    finally:
        if not vista._destruida and consulta is not None:
            vista.after_cancel(consulta)
    assert not vista._ocupada, (
        "La cola de resultados no fue atendida por Tk: "
        f"trabajador activo={vista._hilo.is_alive()}, pendientes={vista._cola.qsize()}."
    )


def test_humo_y_flujo_de_ventana_customtkinter(
    vista: VistaCajero,
    transaccion_cajero: Transaccion,
    plato_sintetico: tuple[np.ndarray, np.ndarray],
) -> None:
    """Comprueba el flujo y tres ciclos de procesar, limpiar y procesar otra foto."""
    vista.update_idletasks()
    assert vista.title() == "Cobro Vision · Cajero"
    assert vista._boton_confirmar.cget("state") == "disabled"
    vista._iniciar_tarea(lambda: transaccion_cajero)
    _esperar_resultado(vista)
    assert vista.transaccion is transaccion_cajero
    assert vista._total.cget("text") == "B/. 4.25"
    assert len(vista._controles_filas) == 6
    assert vista._boton_confirmar.cget("state") == "disabled"
    assert vista._imagen_tk is not None
    vista._alternar_marcas()
    assert not vista._con_marcas
    vista.al_corregir("pollo", "pollo")
    _esperar_resultado(vista)
    assert vista._boton_confirmar.cget("state") == "normal"
    vista.al_quitar("ensalada")
    _esperar_resultado(vista)
    assert vista._total.cget("text") == "B/. 3.50"
    vista.al_agregar("frijoles")
    _esperar_resultado(vista)
    assert vista._total.cget("text") == "B/. 4.25"
    vista._mostrar_explicacion()
    vista.al_confirmar()
    _esperar_resultado(vista)
    assert vista.transaccion.estado == "confirmada"
    assert vista._boton_confirmar.cget("state") == "disabled"
    vista._nueva_transaccion()
    assert vista.transaccion is None and vista._imagen_tk is vista._imagen_vacia
    assert vista._total.cget("text") == "B/. 0.00"
    assert vista._controles_filas == []
    _comprobar_ciclos_de_imagenes(vista, plato_sintetico[0])
    _comprobar_eventos_y_error(vista)
    _comprobar_cierre(vista, transaccion_cajero)


def _comprobar_ciclos_de_imagenes(vista: VistaCajero, imagen: np.ndarray) -> None:
    """Reutiliza el transparente y detecta errores de Tk durante limpiezas repetidas."""
    imagen_vacia = vista._imagen_vacia
    transparente = imagen_vacia.cget("light_image")
    assert transparente.mode == "RGBA" and transparente.size == (1, 1)
    assert transparente.getpixel((0, 0)) == (0, 0, 0, 0)
    otra_imagen = np.ascontiguousarray(imagen[:, ::-1])
    principal = get_ident()

    def crear_pil(*argumentos, **opciones):
        assert get_ident() == principal
        return desde_array(*argumentos, **opciones)

    def crear_ctk(*argumentos, **opciones):
        assert get_ident() == principal
        return tipo_imagen(*argumentos, **opciones)

    from cobro_vision.vista import vista_cajero

    desde_array = vista_cajero.Image.fromarray
    tipo_imagen = vista_cajero.ctk.CTkImage
    with (
        patch.object(
            vista, "report_callback_exception", wraps=vista.report_callback_exception
        ) as errores_tk,
        patch.object(vista_cajero.Image, "fromarray", side_effect=crear_pil),
        patch.object(vista_cajero.ctk, "CTkImage", side_effect=crear_ctk),
    ):
        for _ in range(3):
            for foto in (imagen, otra_imagen):
                vista._iniciar_tarea(partial(vista.controlador.procesar, foto))
                _esperar_resultado(vista)
                assert vista.transaccion.estado in {"procesada", "requiere_revision"}
                assert vista._imagen_tk is not imagen_vacia
                assert vista._foto.cget("image") is vista._imagen_tk
                vista._boton_nueva.invoke()
                gc.collect()
                vista.update()
                assert vista.transaccion is None
                assert vista._imagen_vacia is imagen_vacia
                assert vista._imagen_tk is imagen_vacia
                assert vista._foto.cget("image") is imagen_vacia
                assert vista._foto.cget("text") == "Abra una foto del plato"
        errores_tk.assert_not_called()


@pytest.mark.parametrize("falla", [False, True])
def test_trabajo_no_bloquea_y_todas_las_actualizaciones_ocurren_en_hilo_principal(
    transaccion_cajero: Transaccion, falla: bool
) -> None:
    """Sin Tk: se procesa fuera del hilo principal y se pinta solo al vaciar cola."""
    principal = get_ident()
    iniciado = Event()
    continuar = Event()
    llamadas = []

    def actualizar(*argumentos, **opciones) -> None:
        assert get_ident() == principal, "Un widget fue modificado desde el trabajador."
        llamadas.append((argumentos, opciones))

    ventana = SimpleNamespace(
        _ocupada=False,
        _cerrando=False,
        _cola=Queue(),
        _consulta=None,
        transaccion=None,
        _actualizar_controles=actualizar,
        mostrar_alerta=actualizar,
        mostrar_regiones=actualizar,
        mostrar_recibo=actualizar,
        _tiempos=SimpleNamespace(configure=actualizar),
        after=Mock(side_effect=actualizar),
    )
    ventana._revisar_cola = MethodType(VistaCajero._revisar_cola, ventana)

    def procesar() -> Transaccion:
        assert get_ident() != principal
        iniciado.set()
        assert continuar.wait(3)
        if falla:
            raise ValueError("Foto dañada")
        return transaccion_cajero

    VistaCajero._iniciar_tarea(ventana, procesar)
    try:
        assert iniciado.wait(3)
        assert ventana._ocupada and ventana._cola.empty()
        hilo = ventana._hilo
        VistaCajero._iniciar_tarea(ventana, procesar)
        assert ventana._hilo is hilo  # Los clics repetidos no crean otro trabajador.
        ventana._revisar_cola()
        ventana.after.assert_called_once_with(50, ventana._revisar_cola)
        assert ventana.transaccion is None
    finally:
        continuar.set()
        ventana._hilo.join(3)
    assert not ventana._hilo.is_alive()
    ventana._revisar_cola()
    assert not ventana._ocupada
    if falla:
        assert ventana.transaccion is None
        assert any(args == ("Foto dañada",) for args, _ in llamadas)
    else:
        assert ventana.transaccion is transaccion_cajero


def _comprobar_eventos_y_error(
    vista: VistaCajero,
) -> None:
    """Un callback after se atiende mientras el trabajador sigue ocupado."""
    iniciado = Event()
    continuar = Event()
    atendido = Event()

    def procesar() -> Transaccion:
        iniciado.set()
        assert continuar.wait(3)
        raise RuntimeError("Fallo de captura de prueba")

    vista._iniciar_tarea(procesar)
    try:
        assert iniciado.wait(3)
        vista.after(0, atendido.set)
        vista.update()
        assert atendido.is_set() and vista._ocupada
        assert vista._boton_escanear.cget("state") == "disabled"
    finally:
        continuar.set()
    _esperar_resultado(vista)
    assert "Fallo de captura de prueba" in vista._alerta.cget("text")
    assert vista._boton_escanear.cget("state") == "normal"


def test_al_escanear_archivo_captura_y_cancelacion_sin_abrir_ventana() -> None:
    """El diálogo corre en el evento y el controlador solo al ejecutar la tarea."""
    controlador = Mock()
    ventana = SimpleNamespace(
        _ocupada=False,
        _desde_archivo=True,
        controlador=controlador,
        _nueva_transaccion=Mock(),
        _iniciar_tarea=Mock(),
        mostrar_alerta=Mock(),
    )
    with patch(
        "cobro_vision.vista.vista_cajero.filedialog.askopenfilename", return_value=""
    ):
        VistaCajero.al_escanear(ventana)
    ventana._iniciar_tarea.assert_not_called()
    with patch(
        "cobro_vision.vista.vista_cajero.filedialog.askopenfilename",
        return_value="foto.png",
    ):
        VistaCajero.al_escanear(ventana)
    controlador.cargar_y_procesar.assert_not_called()
    ventana._iniciar_tarea.call_args.args[0]()
    controlador.cargar_y_procesar.assert_called_once()
    ventana._desde_archivo = False
    VistaCajero.al_escanear(ventana)
    assert ventana._iniciar_tarea.call_args.args[0] == controlador.capturar_y_procesar


def _comprobar_cierre(vista: VistaCajero, transaccion_cajero: Transaccion) -> None:
    """Cerrar la ventana no libera la cámara mientras el trabajador la usa."""
    continuar = Event()

    def procesar() -> Transaccion:
        assert continuar.wait(3)
        return transaccion_cajero

    with patch.object(vista.controlador.camara, "liberar") as liberar:
        vista._iniciar_tarea(procesar)
        try:
            vista.destroy()
            assert vista._cerrando and not vista._destruida
            liberar.assert_not_called()
        finally:
            continuar.set()
        _esperar_resultado(vista)
        assert vista._destruida
        liberar.assert_called_once()
