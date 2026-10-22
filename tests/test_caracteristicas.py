"""Descriptores numéricos reproducibles y caché compartida por imagen."""

from itertools import product

import cv2
import numpy as np
import pytest

from cobro_vision.caracteristicas.color import ExtractorHistogramaColor
from cobro_vision.caracteristicas.constructor import ConstructorVector
from cobro_vision.caracteristicas.contexto import ContextoImagen
from cobro_vision.caracteristicas.textura import ExtractorTextura
from cobro_vision.dominio.modelos import Region


def region_prueba(mascara: np.ndarray) -> Region:
    """Construye una región, admitiendo una máscara vacía para probar ceros."""
    filas, columnas = np.nonzero(mascara)
    bbox = (
        (
            int(columnas.min()),
            int(filas.min()),
            int(columnas.max()) + 1,
            int(filas.max()) + 1,
        )
        if mascara.any()
        else (0, 0, 0, 0)
    )
    return Region(1, mascara, bbox, int(mascara.sum()))


def test_contexto_conversiones_y_cache(
    plato_sintetico: tuple[np.ndarray, np.ndarray], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cada representación se reutiliza; LBP distingue radio y número de puntos."""
    imagen, _ = plato_sintetico
    ctx = ContextoImagen(imagen)
    assert ctx.bgr is imagen
    for atributo, conversion in (
        ("hsv", cv2.COLOR_BGR2HSV),
        ("lab", cv2.COLOR_BGR2LAB),
        ("gris", cv2.COLOR_BGR2GRAY),
    ):
        np.testing.assert_array_equal(
            getattr(ctx, atributo), cv2.cvtColor(imagen, conversion)
        )
        assert getattr(ctx, atributo) is getattr(ctx, atributo)
    llamadas = []

    def lbp_simulado(gris: np.ndarray, P: int, R: int, method: str) -> np.ndarray:
        llamadas.append((P, R, method))
        return np.full(gris.shape, R, dtype=np.float64)

    monkeypatch.setattr(
        "cobro_vision.caracteristicas.contexto.local_binary_pattern", lbp_simulado
    )
    primera = ctx.lbp(1, 8)
    assert ctx.lbp(1, 8) is primera
    assert ctx.lbp(2, 8) is not primera
    assert ctx.lbp(1, 16) is not primera
    assert llamadas == [(8, 1, "uniform"), (8, 2, "uniform"), (16, 1, "uniform")]


@pytest.mark.parametrize(
    "espacio,momentos", list(product(("hsv", "lab"), (False, True)))
)
def test_histogramas_mascara_rangos_y_momentos(espacio: str, momentos: bool) -> None:
    """El color exterior no contamina la región y cada canal suma uno."""
    imagen = np.full((20, 30, 3), (255, 0, 0), dtype=np.uint8)
    imagen[5:15, 10:20] = (0, 0, 255)
    mascara = np.zeros(imagen.shape[:2], bool)
    mascara[5:15, 10:20] = True
    extractor = ExtractorHistogramaColor(
        {"espacio": espacio, "bins": [16, 8, 8], "momentos": momentos}
    )
    ctx = ContextoImagen(imagen)
    vector = extractor.extraer(ctx, region_prueba(mascara))
    assert vector.dtype == np.float32 and vector.shape == (38 if momentos else 32,)
    assert len(extractor.nombres()) == vector.size
    for inicio, fin in ((0, 16), (16, 24), (24, 32)):
        assert vector[inicio:fin].sum() == pytest.approx(1)
        assert np.count_nonzero(vector[inicio:fin]) == 1
    if espacio == "hsv":
        assert vector[0] == 1  # Rojo: H=0. Azul sería otro bin.
        assert vector[23] == 1 and vector[31] == 1  # S=V=255, rango completo.
    if momentos:
        valores = getattr(ctx, espacio)[10, 10].astype(float)
        rangos = np.array((180, 256, 256) if espacio == "hsv" else (256, 256, 256))
        np.testing.assert_allclose(vector[32::2], valores / rangos)
        assert np.all(vector[33::2] == 0)
    np.testing.assert_array_equal(
        vector, extractor.extraer(ctx, region_prueba(mascara))
    )


@pytest.mark.parametrize("gradiente,glcm", list(product((False, True), repeat=2)))
def test_textura_dimension_normalizacion_y_sin_nan(
    datos_config: dict,
    plato_sintetico: tuple[np.ndarray, np.ndarray],
    gradiente: bool,
    glcm: bool,
) -> None:
    """Todas las opciones tienen dimensión fija y conservan los valores al repetirse."""
    imagen, mapa = plato_sintetico
    config = datos_config["caracteristicas"][1]
    config.update(gradiente=gradiente, glcm=glcm)
    extractor = ExtractorTextura(config)
    region = region_prueba(mapa == 4)
    ctx = ContextoImagen(imagen)
    vector = extractor.extraer(ctx, region)
    dimension = 10 + 3 * gradiente + 4 * glcm
    assert vector.shape == (dimension,) and vector.dtype == np.float32
    assert len(extractor.nombres()) == dimension
    assert np.isfinite(vector).all()
    assert vector[:10].sum() == pytest.approx(1)
    np.testing.assert_array_equal(vector, extractor.extraer(ctx, region))


def test_textura_distingue_rayas(datos_config: dict) -> None:
    """El gradiente y GLCM detectan cambios espaciales ausentes en un área uniforme."""
    lisa = np.full((32, 32, 3), 100, np.uint8)
    rayada = lisa.copy()
    rayada[:, ::4] = 200
    config = datos_config["caracteristicas"][1]
    config.update(gradiente=True, glcm=True)
    extractor = ExtractorTextura(config)
    region = region_prueba(np.ones((32, 32), bool))
    vector_liso = extractor.extraer(ContextoImagen(lisa), region)
    vector_rayado = extractor.extraer(ContextoImagen(rayada), region)
    assert vector_liso[10] == 0 < vector_rayado[10]
    assert vector_liso[13] == 0 < vector_rayado[13]
    assert np.isfinite(vector_liso).all()  # Correlación de imagen constante.


def test_glcm_rellena_exterior_con_media(datos_config: dict) -> None:
    """Los huecos de máscara usan la media y no el fondo negro del bbox."""
    imagen = np.zeros((10, 10, 3), np.uint8)
    mascara = np.zeros((10, 10), bool)
    mascara[2:8, 2:8] = True
    mascara[4:6, 4:6] = False
    imagen[mascara] = 128
    config = datos_config["caracteristicas"][1]
    config.update(gradiente=False, glcm=True)
    vector = ExtractorTextura(config).extraer(
        ContextoImagen(imagen), region_prueba(mascara)
    )
    np.testing.assert_allclose(vector[-4:], (0, 1, 1, 1))


@pytest.mark.parametrize("puntos", [8, 16])
def test_region_vacia_y_un_pixel(datos_config: dict, puntos: int) -> None:
    """Una región vacía da ceros; una región de un píxel tampoco genera NaN."""
    color, textura = datos_config["caracteristicas"]
    textura["lbp"]["puntos"] = puntos
    textura["glcm"] = True
    constructor = ConstructorVector(
        [ExtractorHistogramaColor(color), ExtractorTextura(textura)]
    )
    imagen = np.zeros((10, 10, 3), dtype=np.uint8)
    vacia = region_prueba(np.zeros((10, 10), bool))
    vector = constructor.construir(ContextoImagen(imagen), vacia)
    assert vector.shape == (38 + puntos + 2 + 3 + 4,)
    assert not vector.any()
    mascara = np.zeros((10, 10), bool)
    mascara[5, 5] = True
    vector = constructor.construir(ContextoImagen(imagen), region_prueba(mascara))
    assert np.isfinite(vector).all()


def test_constructor_orden_dimension_lote_y_contexto_unico(
    datos_config: dict,
    plato_sintetico: tuple[np.ndarray, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """El lote reutiliza contexto, mantiene orden y almacena cada vector de región."""
    imagen, mapa = plato_sintetico
    color, textura = datos_config["caracteristicas"]
    extractores = [ExtractorHistogramaColor(color), ExtractorTextura(textura)]
    constructor = ConstructorVector(extractores)
    assert constructor.dimension() == 51
    assert constructor.nombres() == extractores[0].nombres() + extractores[1].nombres()
    assert len(set(constructor.nombres())) == constructor.dimension()
    regiones = [region_prueba(mapa == indice) for indice in (1, 2, 3, 4)]
    contextos = []

    def crear_contexto(bgr: np.ndarray) -> ContextoImagen:
        ctx = ContextoImagen(bgr)
        contextos.append(ctx)
        return ctx

    monkeypatch.setattr(
        "cobro_vision.caracteristicas.constructor.ContextoImagen", crear_contexto
    )
    vectores = constructor.construir_lote(imagen, regiones)
    assert len(contextos) == 1
    assert vectores.shape == (4, constructor.dimension())
    assert vectores.dtype == np.float32 and np.isfinite(vectores).all()
    for region, vector in zip(regiones, vectores):
        np.testing.assert_array_equal(region.vector, vector)
        np.testing.assert_array_equal(
            vector, constructor.construir(contextos[0], region)
        )
    np.testing.assert_array_equal(
        vectores, constructor.construir_lote(imagen, regiones)
    )
    assert constructor.construir_lote(imagen, []).shape == (0, 51)
    assert ConstructorVector([]).construir_lote(imagen, regiones).shape == (4, 0)
