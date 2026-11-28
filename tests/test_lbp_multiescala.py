from copy import deepcopy

import numpy as np
import pytest

from cobro_vision.caracteristicas.contexto import ContextoImagen
from cobro_vision.caracteristicas.textura import ExtractorTextura
from cobro_vision.dominio.modelos import Region


@pytest.mark.parametrize(
    "escalas,dimension",
    [
        ([{"radio": 1, "puntos": 8}], 17),
        ([{"radio": 2, "puntos": 16}], 25),
        ([
            {"radio": 1, "puntos": 8},
            {"radio": 2, "puntos": 16},
        ], 35),
        ([
            {"radio": 1, "puntos": 8},
            {"radio": 2, "puntos": 16},
            {"radio": 3, "puntos": 24},
        ], 61),
    ],
)
def test_lbp_multiescala(escalas, dimension):
    config = {
        "lbp": {"radio": 1, "puntos": 8},
        "lbp_multiescala": escalas,
        "gradiente": True,
        "glcm": True,
    }

    rng = np.random.default_rng(44)

    imagen = rng.integers(
        0, 256, size=(40, 40, 3), dtype=np.uint8
    )

    mascara = np.ones((40, 40), dtype=bool)

    region = Region(
        1, mascara, (0, 0, 40, 40), 1600
    )

    extractor = ExtractorTextura(config)
    contexto = ContextoImagen(imagen)

    vector = extractor.extraer(contexto, region)

    assert vector.shape == (dimension,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()

    nombres = extractor.nombres()

    assert len(nombres) == dimension
    assert len(set(nombres)) == dimension

    inicio = 0

    for escala in escalas:
        fin = inicio + escala["puntos"] + 2

        assert vector[inicio:fin].sum() == pytest.approx(1)

        inicio = fin

    np.testing.assert_array_equal(
        vector, extractor.extraer(contexto, region)
    )

    vacia = Region(
        2,
        np.zeros((40, 40), dtype=bool),
        (0, 0, 0, 0),
        0,
    )

    vector_vacio = extractor.extraer(contexto, vacia)

    assert vector_vacio.shape == (dimension,)
    assert not vector_vacio.any()


def test_compatibilidad_lbp_original():
    config = {
        "lbp": {"radio": 1, "puntos": 8},
        "gradiente": True,
        "glcm": True,
    }

    config_nueva = deepcopy(config)

    config_nueva["lbp_multiescala"] = [
        {"radio": 1, "puntos": 8}
    ]

    rng = np.random.default_rng(44)

    imagen = rng.integers(
        0, 256, size=(40, 40, 3), dtype=np.uint8
    )

    mascara = np.ones((40, 40), dtype=bool)

    region = Region(
        1, mascara, (0, 0, 40, 40), 1600
    )

    contexto = ContextoImagen(imagen)

    original = ExtractorTextura(config)
    nuevo = ExtractorTextura(config_nueva)

    np.testing.assert_array_equal(
        original.extraer(contexto, region),
        nuevo.extraer(contexto, region),
    )

    assert original.nombres()[0] == "lbp_00"
    assert nuevo.nombres()[0] == "lbp_r1_p8_00"

    assert original.nombres()[-7:] == nuevo.nombres()[-7:]
