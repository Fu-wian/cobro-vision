import numpy as np
import pytest

from cobro_vision.dominio.errores import PlatoNoDetectado
from cobro_vision.preprocesamiento.detector_plato import DetectorPlato


def configuracion_e06(datos_config, metodo="hibrido"):
    return {
        **datos_config["plato"],
        "metodo": metodo,
        "radio_max_frac": 0.75,
        "reducir_radio": 1.0,
        "area_minima_frac": 0.10,
        "area_min_hough_frac": 0.30,
        "umbral": {
            "s_max": 60,
            "v_min": 140,
        },
        "umbral_respaldo": {
            "s_max": 80,
            "v_min": 120,
        },
        "adaptativo": {
            "elongacion_minima": 1.35,
            "saturacion_minima": 45,
            "densidad_minima": 0.15,
            "ratio_maximo_tapa": 0.35,
            "retencion_color_minima": 0.95,
            "margen_corte_frac": 0.05,
        },
    }


@pytest.mark.parametrize(
    "area,esperado",
    [
        (20, 60),
        (29, 60),
        (30, 30),
        (31, 31),
        (40, 40),
    ],
)
def test_seleccion_hibrida(
    datos_config, monkeypatch, area, esperado
):
    imagen = np.zeros(
        (100, 100, 3), dtype=np.uint8
    )

    principal = np.zeros(
        (100, 100), dtype=bool
    )

    principal[:area, :] = True

    respaldo = np.zeros_like(principal)
    respaldo[:60, :] = True

    monkeypatch.setattr(
        DetectorPlato,
        "_hough",
        lambda self, imagen: principal,
    )

    monkeypatch.setattr(
        DetectorPlato,
        "_umbral",
        lambda self, imagen: respaldo,
    )

    detector = DetectorPlato(
        configuracion_e06(datos_config)
    )

    resultado = detector.detectar(imagen)

    assert resultado.dtype == np.bool_
    assert resultado.sum() == esperado * 100


def test_hibrido_conserva_respaldo_original(
    datos_config, monkeypatch
):
    imagen = np.zeros(
        (100, 100, 3), dtype=np.uint8
    )

    llamadas = []

    def umbral(self, imagen):
        llamadas.append(
            self._config["umbral"]["s_max"]
        )

        mascara = np.zeros(
            (100, 100), dtype=bool
        )

        mascara[:40, :] = True

        return mascara

    monkeypatch.setattr(
        DetectorPlato,
        "_hough",
        lambda self, imagen: None,
    )

    monkeypatch.setattr(
        DetectorPlato, "_umbral", umbral
    )

    detector = DetectorPlato(
        configuracion_e06(datos_config)
    )

    resultado = detector.detectar(imagen)

    assert resultado.sum() == 4000
    assert llamadas == [60]


def test_hibrido_respaldo_flexible(
    datos_config, monkeypatch
):
    imagen = np.zeros(
        (100, 100, 3), dtype=np.uint8
    )

    llamadas = []

    def umbral(self, imagen):
        s_max = self._config["umbral"]["s_max"]

        llamadas.append(s_max)

        mascara = np.zeros(
            (100, 100), dtype=bool
        )

        if s_max == 60:
            mascara[:20, :] = True
        else:
            mascara[:60, :] = True

        return mascara

    monkeypatch.setattr(
        DetectorPlato,
        "_hough",
        lambda self, imagen: None,
    )

    monkeypatch.setattr(
        DetectorPlato, "_umbral", umbral
    )

    detector = DetectorPlato(
        configuracion_e06(datos_config)
    )

    assert detector.detectar(imagen).sum() == 6000
    assert llamadas == [60, 80]


def test_hibrido_sin_plato(
    datos_config, monkeypatch
):
    imagen = np.zeros(
        (100, 100, 3), dtype=np.uint8
    )

    monkeypatch.setattr(
        DetectorPlato,
        "_hough",
        lambda self, imagen: None,
    )

    monkeypatch.setattr(
        DetectorPlato,
        "_umbral",
        lambda self, imagen: np.zeros(
            (100, 100), dtype=bool
        ),
    )

    detector = DetectorPlato(
        configuracion_e06(datos_config)
    )

    with pytest.raises(PlatoNoDetectado):
        detector.detectar(imagen)


@pytest.mark.parametrize("lado", ["izquierda", "derecha"])
def test_adaptativo_separa_tapa(
    datos_config, lado
):
    imagen = np.full(
        (200, 400, 3), 30, dtype=np.uint8
    )

    mascara = np.zeros(
        (200, 400), dtype=bool
    )

    mascara[40:160, 40:360] = True

    imagen[mascara] = (230, 230, 230)

    if lado == "izquierda":
        imagen[60:140, 60:180] = (20, 100, 200)
    else:
        imagen[60:140, 220:340] = (20, 100, 200)

    detector = DetectorPlato(
        configuracion_e06(
            datos_config, "adaptativo"
        )
    )

    resultado = detector._separar_tapa(
        imagen, mascara
    )

    assert resultado.sum() < mascara.sum()

    assert not np.any(
        resultado & ~mascara
    )

    if lado == "izquierda":
        assert resultado[100, 100]
        assert not resultado[100, 300]
    else:
        assert resultado[100, 300]
        assert not resultado[100, 100]


def test_adaptativo_conserva_mascara_ambigua(
    datos_config
):
    imagen = np.full(
        (200, 400, 3), 30, dtype=np.uint8
    )

    mascara = np.zeros(
        (200, 400), dtype=bool
    )

    mascara[40:160, 40:360] = True

    imagen[mascara] = (20, 100, 200)

    detector = DetectorPlato(
        configuracion_e06(
            datos_config, "adaptativo"
        )
    )

    resultado = detector._separar_tapa(
        imagen, mascara
    )

    np.testing.assert_array_equal(
        resultado, mascara
    )


def test_adaptativo_conserva_mascara_circular(
    datos_config
):
    imagen = np.zeros(
        (200, 200, 3), dtype=np.uint8
    )

    filas, columnas = np.indices(
        (200, 200)
    )

    mascara = (
        (filas - 100) ** 2
        + (columnas - 100) ** 2
        <= 80 ** 2
    )

    imagen[mascara] = (20, 100, 200)

    detector = DetectorPlato(
        configuracion_e06(
            datos_config, "adaptativo"
        )
    )

    resultado = detector._separar_tapa(
        imagen, mascara
    )

    np.testing.assert_array_equal(
        resultado, mascara
    )
