"""Pruebas del esquema completo y sus huellas reproducibles."""

import re
from pathlib import Path

import pytest
import yaml

from cobro_vision.configuracion import Configuracion
from cobro_vision.dominio.errores import ConfiguracionInvalida


def escribir_config(tmp_path: Path, datos: dict) -> Path:
    """Guarda una variante de la configuración de ejemplo."""
    ruta = tmp_path / "config.yaml"
    ruta.write_text(yaml.safe_dump(datos, allow_unicode=True), encoding="utf-8")
    return ruta


def test_cargar_configuracion_y_copias(ruta_proyecto: Path) -> None:
    """El ejemplo se carga y sus secciones no permiten cambios accidentales."""
    config = Configuracion.cargar(ruta_proyecto / "config.yaml")
    assert config.semilla == 42
    assert config.seccion("imagen")["lado_mayor"] == 640
    assert config.seccion("caracteristicas")["extractores"][0]["espacio"] == "hsv"
    imagen = config.seccion("imagen")
    imagen["lado_mayor"] = 20
    imagen["filtro"]["tipo"] = "ninguno"
    assert config.seccion("imagen")["lado_mayor"] == 640
    assert config.seccion("imagen")["filtro"]["tipo"] == "mediana"
    with pytest.raises(ConfiguracionInvalida, match="inexistente"):
        config.seccion("inexistente")


def test_hash_estable_y_selectivo(tmp_path: Path, datos_config: dict) -> None:
    """El hash ignora el orden de claves y solo cambia por las secciones elegidas."""
    config = Configuracion.cargar(escribir_config(tmp_path, datos_config))
    huella = config.hash()
    assert re.fullmatch("[0-9a-f]{12}", huella)
    reordenado = dict(reversed(list(datos_config.items())))
    assert Configuracion.cargar(escribir_config(tmp_path, reordenado)).hash() == huella
    vision = ["imagen", "plato", "segmentacion", "caracteristicas"]
    assert config.hash(vision) == config.hash(list(reversed(vision)))
    datos_config["ui"]["tema"] = "dark"
    cambiado = Configuracion.cargar(escribir_config(tmp_path, datos_config))
    assert cambiado.hash() != huella
    assert cambiado.hash(vision) == config.hash(vision)
    datos_config["imagen"]["lado_mayor"] = 320
    assert Configuracion.cargar(escribir_config(tmp_path, datos_config)).hash(
        vision
    ) != config.hash(vision)
    with pytest.raises(ConfiguracionInvalida, match="ausente"):
        config.hash(["ausente"])


@pytest.mark.parametrize(
    ("campo", "valor"),
    [
        ("semilla", True),
        ("semilla", -1),
        ("semilla", 2**32),
        ("semilla", 10**400),
        ("rutas.catalogo", 3),
        ("rutas.imagenes", ""),
        ("imagen.lado_mayor", 0),
        ("imagen.filtro.tamano", 2),
        ("imagen.filtro.tipo", "otro"),
        ("imagen.correccion_color.metodo", "otro"),
        ("plato.metodo", "otro"),
        ("plato.radio_max_frac", 0.1),
        ("plato.area_minima_frac", 1.1),
        ("plato.umbral.s_max", 256),
        ("segmentacion.clase", "otro"),
        ("segmentacion.n_segmentos", 0),
        ("segmentacion.compacidad", 0),
        ("segmentacion.sigma", float("nan")),
        ("segmentacion.ventanas.paso", 0),
        ("clasificador.tipo", "otro"),
        ("clasificador.kernel", "otro"),
        ("clasificador.C", -1),
        ("clasificador.gamma", "desconocido"),
        ("clasificador.gamma", True),
        ("clasificador.gamma", 0),
        ("clasificador.umbral_confianza", -1),
        ("clasificador.busqueda.folds", 1),
        ("clasificador.busqueda.C", []),
        ("clasificador.busqueda.gamma", ["otro"]),
        ("clasificador.class_weight", "otro"),
        ("entrenamiento.modo", "otro"),
        ("entrenamiento.pureza_minima", 0),
        ("posproceso.max_proteinas", -1),
        ("posproceso.clase_minima_frac", 2),
        ("cobro.politica", "otro"),
        ("cobro.por_plato.precios_por_proteina_centavos", {"pollo": 450.0}),
        ("cobro.por_plato.precios_por_proteina_centavos", {"pollo": True}),
        ("cobro.por_plato.precios_por_proteina_centavos", {"pollo": -1}),
        ("camara.clase", "otra"),
        ("camara.indice", -1),
        ("ui.exigir_revision", "true"),
        ("ui.tema", "otro"),
        ("ui.tamano_ventana", [0, 720]),
        ("ui.tamano_ventana", [1200]),
        ("caracteristicas", []),
    ],
)
def test_valores_invalidos_con_nombre_del_campo(
    tmp_path: Path, datos_config: dict, campo: str, valor: object
) -> None:
    """Los valores inválidos señalan el campo en lugar de fallar después."""
    destino = datos_config
    partes = campo.split(".")
    for parte in partes[:-1]:
        destino = destino[parte]
    destino[partes[-1]] = valor
    with pytest.raises(ConfiguracionInvalida, match=re.escape(campo)):
        Configuracion.cargar(escribir_config(tmp_path, datos_config))


@pytest.mark.parametrize(
    ("indice", "campo", "valor"),
    [
        (0, "clase", "Otra"),
        (0, "espacio", "rgb"),
        (0, "bins", [16, 8]),
        (0, "bins", [16, 0, 8]),
        (0, "momentos", 1),
        (1, "gradiente", "si"),
        (1, "glcm", None),
        (1, "glcm_distancias", []),
        (1, "glcm_distancias", [0]),
        (1, "glcm_distancias", [1.5]),
        (1, "glcm_angulos", []),
        (1, "glcm_angulos", [float("nan")]),
    ],
)
def test_extractores_invalidos(
    tmp_path: Path, datos_config: dict, indice: int, campo: str, valor: object
) -> None:
    """Cada extractor valida sus parámetros, sin crear componentes de visión."""
    datos_config["caracteristicas"][indice][campo] = valor
    with pytest.raises(ConfiguracionInvalida, match=campo):
        Configuracion.cargar(escribir_config(tmp_path, datos_config))


@pytest.mark.parametrize("contenido", ["", "[]", "imagen: [", "semilla: 42"])
def test_yaml_incompleto_o_malformado(tmp_path: Path, contenido: str) -> None:
    """Los documentos inválidos se convierten a ConfiguracionInvalida."""
    ruta = tmp_path / "invalida.yaml"
    ruta.write_text(contenido, encoding="utf-8")
    with pytest.raises(ConfiguracionInvalida):
        Configuracion.cargar(ruta)


def test_variantes_validas_y_roi(tmp_path: Path, datos_config: dict) -> None:
    """Se admiten las variantes del esquema y se exige el ROI de tarjeta gris."""
    datos_config["posproceso"]["max_proteinas"] = None
    datos_config["clasificador"]["gamma"] = 0.1
    datos_config["clasificador"]["class_weight"] = {"pollo": 2}
    datos_config["imagen"]["correccion_color"] = {"metodo": "tarjeta_gris"}
    with pytest.raises(ConfiguracionInvalida, match="roi"):
        Configuracion.cargar(escribir_config(tmp_path, datos_config))
    datos_config["imagen"]["correccion_color"]["roi"] = [0, 0, 20, 20]
    config = Configuracion.cargar(escribir_config(tmp_path, datos_config))
    assert config.seccion("imagen")["correccion_color"]["roi"] == [0, 0, 20, 20]
