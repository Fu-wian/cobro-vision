"""Recursos compartidos por las pruebas del repositorio."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from helpers import generar_plato_sintetico

from cobro_vision.negocio.catalogo import CatalogoAlimentos


@pytest.fixture
def plato_sintetico(datos_config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Crea un plato artificial con la semilla compartida del YAML."""
    return generar_plato_sintetico(datos_config["semilla"], 4)


@pytest.fixture
def ruta_proyecto() -> Path:
    """Devuelve la raíz del repositorio sin depender del directorio de trabajo."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def datos_config(ruta_proyecto: Path) -> dict:
    """Carga los valores de ejemplo para probar variaciones aisladas."""
    return yaml.safe_load((ruta_proyecto / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def ruta_config_temporal(
    tmp_path: Path, ruta_proyecto: Path, datos_config: dict
) -> Path:
    """Escribe una configuración cuyo catálogo se genera en una carpeta temporal."""
    datos_config["rutas"]["catalogo"] = str(tmp_path / "datos" / "catalogo.db")
    datos_config["rutas"]["semilla_catalogo"] = str(
        ruta_proyecto / "data/catalogo_seed.csv"
    )
    ruta = tmp_path / "config.yaml"
    ruta.write_text(yaml.safe_dump(datos_config, allow_unicode=True), encoding="utf-8")
    return ruta


@pytest.fixture
def catalogo(tmp_path: Path, ruta_proyecto: Path) -> CatalogoAlimentos:
    """Inicializa un catálogo de ejemplo sin modificar los datos del proyecto."""
    catalogo = CatalogoAlimentos(tmp_path / "datos" / "catalogo.db")
    catalogo.inicializar(ruta_proyecto / "data/catalogo_seed.csv")
    return catalogo
