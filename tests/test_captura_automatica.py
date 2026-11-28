import numpy as np

from cobro_vision.adquisicion.captura_automatica import (
    CapturaAutomatica,
)


def imagenes():
    fondo = np.zeros((120, 160, 3), dtype=np.uint8)

    plato = fondo.copy()
    plato[20:100, 30:130] = 220

    return fondo, plato


def test_superficie_vacia():

    fondo, _ = imagenes()

    captura = CapturaAutomatica()
    captura.calibrar(fondo)

    assert not captura.actualizar(fondo, 0)
    assert not captura.actualizar(fondo, 3)
    assert captura.estado == "esperando"


def test_captura_plato_estable():

    fondo, plato = imagenes()

    captura = CapturaAutomatica()
    captura.calibrar(fondo)

    assert not captura.actualizar(plato, 0)
    assert not captura.actualizar(plato, 1)
    assert captura.actualizar(plato, 2)
    assert captura.estado == "bloqueado"


def test_evitar_capturas_duplicadas():

    fondo, plato = imagenes()

    captura = CapturaAutomatica()
    captura.calibrar(fondo)

    captura.actualizar(plato, 0)

    assert captura.actualizar(plato, 2)
    assert not captura.actualizar(plato, 4)
    assert not captura.actualizar(plato, 8)


def test_retiro_y_nuevo_plato():

    fondo, plato = imagenes()

    captura = CapturaAutomatica()
    captura.calibrar(fondo)

    captura.actualizar(plato, 0)
    assert captura.actualizar(plato, 2)

    captura.esperar_retiro()

    assert not captura.actualizar(plato, 3)
    assert not captura.actualizar(fondo, 4)
    assert not captura.actualizar(fondo, 5)

    assert captura.estado == "esperando"

    assert not captura.actualizar(plato, 6)
    assert captura.actualizar(plato, 8)


def test_movimiento_reinicia_estabilidad():

    fondo, plato = imagenes()

    captura = CapturaAutomatica()
    captura.calibrar(fondo)

    captura.actualizar(plato, 0)

    movido = np.roll(plato, 25, axis=1)

    assert not captura.actualizar(movido, 1)
    assert not captura.actualizar(movido, 2)
    assert captura.actualizar(movido, 3)


def test_sin_calibracion_no_captura():

    _, plato = imagenes()

    captura = CapturaAutomatica()

    assert not captura.actualizar(plato, 0)
    assert not captura.actualizar(plato, 5)


def test_reinicio_manual_permite_recapturar():
    import numpy as np
    from cobro_vision.adquisicion.captura_automatica import CapturaAutomatica

    fondo = np.zeros((120, 160, 3), dtype=np.uint8)
    plato = fondo.copy()
    plato[20:100, 30:130] = 220

    captura = CapturaAutomatica(estabilidad_s=0.8)
    captura.calibrar(fondo)

    assert not captura.actualizar(plato, ahora=0.0)
    assert captura.actualizar(plato, ahora=1.0)
    assert captura.estado == "bloqueado"

    assert not captura.actualizar(plato, ahora=2.0)

    captura.reiniciar()

    assert captura.estado == "esperando"
    assert not captura.actualizar(plato, ahora=3.0)
    assert captura.actualizar(plato, ahora=4.0)
    assert captura.estado == "bloqueado"
