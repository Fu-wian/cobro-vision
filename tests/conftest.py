"""Recursos compartidos por las pruebas del repositorio."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from helpers import generar_plato_sintetico

from cobro_vision.configuracion import Configuracion
from cobro_vision.controlador.controlador import ControladorTransaccion
from cobro_vision.controlador.fabrica import FabricaPipeline
from cobro_vision.dominio.modelos import Region, ResultadoClasificacion, Transaccion
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


@pytest.fixture
def controlador_cajero(
    ruta_config_temporal: Path, tmp_path: Path
) -> ControladorTransaccion:
    """Aísla el catálogo y las transacciones CSV del cajero en temporales."""
    datos = yaml.safe_load(ruta_config_temporal.read_text(encoding="utf-8"))
    datos["rutas"]["salidas"] = str(tmp_path / "salidas")
    ruta_config_temporal.write_text(yaml.safe_dump(datos), encoding="utf-8")
    return FabricaPipeline.crear(Configuracion.cargar(ruta_config_temporal))


@pytest.fixture
def transaccion_cajero(controlador_cajero: ControladorTransaccion) -> Transaccion:
    """Crea cuatro regiones artificiales, dos de pollo pendientes de revisión."""
    imagen = np.full((40, 80, 3), 200, dtype=np.uint8)
    transaccion = Transaccion(
        "T-20261021-120000-abcd",
        imagen,
        imagen_procesada=imagen.copy(),
        mascara_plato=np.ones((40, 80), dtype=bool),
    )
    for indice, clase in enumerate(["pollo", "pollo", "arroz_amarillo", "ensalada"]):
        mascara = np.zeros((40, 80), dtype=bool)
        mascara[:, indice * 20 : (indice + 1) * 20] = True
        revision = clase == "pollo"
        transaccion.regiones.append(
            Region(
                indice + 1,
                mascara,
                (indice * 20, 0, (indice + 1) * 20, 40),
                800,
                resultado=ResultadoClasificacion(clase, 0.2, revision, "pescado_frito"),
                estado="revisar" if revision else "aceptada",
            )
        )
    transaccion.recibo = controlador_cajero.calculadora.calcular(transaccion.regiones)
    transaccion.estado = "requiere_revision"
    transaccion.tiempos = {"cobro": 0.5, "total": 12.25}
    return transaccion
