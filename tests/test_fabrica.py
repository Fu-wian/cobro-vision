"""Selección centralizada de variantes sin cargar modelos entrenados."""

from pathlib import Path

import pytest

from cobro_vision.adquisicion.camara import CamaraArchivo
from cobro_vision.clasificacion.falso import ClasificadorFalso
from cobro_vision.configuracion import Configuracion
from cobro_vision.controlador.fabrica import FabricaPipeline
from cobro_vision.dominio.errores import ConfiguracionInvalida
from cobro_vision.negocio.catalogo import CatalogoAlimentos
from cobro_vision.segmentacion.superpixeles import SegmentadorSuperpixeles
from cobro_vision.segmentacion.ventanas import SegmentadorVentanas


def test_fabrica_por_defecto_falso_y_catalogo_sin_sobrescribir(
    ruta_config_temporal: Path,
) -> None:
    """Inicializa un catálogo ausente y conserva precios modificados al crear otra vez."""
    config = Configuracion.cargar(ruta_config_temporal)
    controlador = FabricaPipeline.crear(config)
    assert isinstance(controlador.camara, CamaraArchivo)
    assert isinstance(controlador.segmentador, SegmentadorSuperpixeles)
    assert isinstance(controlador.clasificador, ClasificadorFalso)
    assert (
        controlador.clasificador.nombres_caracteristicas
        == controlador.constructor.nombres()
    )
    catalogo = CatalogoAlimentos(Path(config.seccion("rutas")["catalogo"]))
    catalogo.actualizar_precio("pollo", 321)
    FabricaPipeline.crear(config)
    assert catalogo.obtener("pollo").precio_centavos == 321


def test_fabrica_ventanas_orden_extractores_y_cobro_por_plato(
    ruta_config_temporal: Path,
) -> None:
    """Las opciones del YAML determinan el segmentador, vector y política reales."""
    datos = Configuracion.cargar(ruta_config_temporal)._datos.copy()
    datos["segmentacion"] = {**datos["segmentacion"], "clase": "SegmentadorVentanas"}
    datos["caracteristicas"] = list(reversed(datos["caracteristicas"]))
    datos["cobro"] = {
        "politica": "CobroPorPlato",
        "por_plato": {"precios_por_proteina_centavos": {"pollo": 450}},
    }
    controlador = FabricaPipeline.crear(Configuracion(datos))
    assert isinstance(controlador.segmentador, SegmentadorVentanas)
    assert controlador.constructor.nombres()[0].startswith("lbp_")
    from cobro_vision.dominio.modelos import ItemRecibo

    catalogo = CatalogoAlimentos(Path(datos["rutas"]["catalogo"]))
    recibo = controlador.calculadora._politica.calcular(
        [ItemRecibo(catalogo.obtener("pollo"), "auto", [], 0.1, False, 0)]
    )
    assert recibo.total_centavos == 450


@pytest.mark.parametrize(
    "campo", ["camara", "segmentacion", "caracteristicas", "clasificador", "cobro"]
)
def test_fabrica_rechaza_nombres_no_registrados(
    ruta_config_temporal: Path, campo: str
) -> None:
    """El registro rechaza también configuraciones construidas sin pasar por cargar."""
    datos = Configuracion.cargar(ruta_config_temporal)._datos
    if campo == "caracteristicas":
        datos[campo][0]["clase"] = "NoExiste"
    else:
        clave = (
            "tipo"
            if campo == "clasificador"
            else "politica"
            if campo == "cobro"
            else "clase"
        )
        datos[campo][clave] = "NoExiste"
    with pytest.raises(ConfiguracionInvalida, match=campo):
        FabricaPipeline.crear(Configuracion(datos))


def test_svm_informa_etapa_pendiente(ruta_config_temporal: Path) -> None:
    """La Etapa 2 no intenta entrenar ni descargar o cargar un SVM."""
    datos = Configuracion.cargar(ruta_config_temporal)._datos
    datos["clasificador"]["tipo"] = "svm"
    with pytest.raises(ConfiguracionInvalida, match="etapa 5"):
        FabricaPipeline.crear(Configuracion(datos))
