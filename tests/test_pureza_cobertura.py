from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from cobro_vision.clasificacion.instancias import ConstructorInstancias
from cobro_vision.configuracion import Configuracion
from cobro_vision.dominio.errores import ConfiguracionInvalida


def crear_constructor(valores, experimental=True):

    mapa = np.array(valores, dtype=np.int32).reshape(10, 10)

    mascara = np.ones((10, 10), dtype=bool)

    region = SimpleNamespace(
        mascara=mascara,
        area=100,
    )

    config = {"pureza_minima": 0.8}

    if experimental:
        config["seleccion_regiones"] = {
            "pureza_etiquetada_minima": 0.95,
            "cobertura_minima": 0.50,
        }

    constructor = ConstructorInstancias(
        config=SimpleNamespace(
            seccion=lambda nombre: config
        ),
        preprocesador=None,
        detector=None,
        segmentador=SimpleNamespace(
            segmentar=lambda imagen, mascara: [region]
        ),
        constructor=None,
        mapa=SimpleNamespace(
            clase_de=lambda indice: {
                1: "pollo",
                2: "menestra"
            }.get(indice)
        ),
    )

    return constructor, mapa, mascara


@pytest.mark.parametrize(
    "valores, entrenamiento, experimental, esperado",
    [
        ([1]*100, True, True, 1),
        ([0]*100, True, True, 0),
        ([1]*60 + [0]*40, True, True, 1),
        ([1]*50 + [2]*10 + [0]*40, True, True, 0),
        ([1]*40 + [0]*60, True, True, 0),
        ([1]*60 + [0]*40, False, True, 0),
        ([1]*60 + [0]*40, True, False, 0),
        ([1]*80 + [0]*20, False, True, 1),
        ([1]*50 + [0]*50, True, True, 1),
        ([1]*57 + [2]*3 + [0]*40, True, True, 1),
    ],
)
def test_seleccion_regiones(
    valores, entrenamiento, experimental, esperado
):
    constructor, mapa, mascara = crear_constructor(
        valores, experimental
    )

    imagen = np.zeros((10, 10, 3), dtype=np.uint8)

    filas = constructor._regiones(
        imagen,
        mascara,
        mapa,
        "prueba",
        para_entrenamiento=entrenamiento,
    )

    assert len(filas) == esperado

    if esperado:
        assert filas[0][1] == "pollo"


@pytest.mark.parametrize(
    "pureza, cobertura",
    [
        (0, 0.5),
        (1.1, 0.5),
        (0.95, 0),
        (0.95, 1.1),
        (True, 0.5),
        (0.95, "incorrecto"),
    ],
)
def test_configuracion_invalida(
    tmp_path, pureza, cobertura
):
    from pathlib import Path

    config = yaml.safe_load(
        Path("config.yaml").read_text(encoding="utf-8-sig")
    )

    config["entrenamiento"]["seleccion_regiones"] = {
        "pureza_etiquetada_minima": pureza,
        "cobertura_minima": cobertura,
    }

    ruta = tmp_path / "config.yaml"
    ruta.write_text(
        yaml.safe_dump(config),
        encoding="utf-8"
    )

    with pytest.raises(ConfiguracionInvalida):
        Configuracion.cargar(ruta)
