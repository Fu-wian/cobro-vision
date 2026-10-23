"""Selección centralizada y explícita de los componentes del pipeline."""

from pathlib import Path
from typing import ClassVar

from cobro_vision.adquisicion.camara import CamaraArchivo, CamaraUSB
from cobro_vision.adquisicion.mapa_etiquetas import MapaEtiquetas
from cobro_vision.caracteristicas.color import ExtractorHistogramaColor
from cobro_vision.caracteristicas.constructor import ConstructorVector
from cobro_vision.caracteristicas.textura import ExtractorTextura
from cobro_vision.clasificacion.entrenador import EntrenadorModelo
from cobro_vision.clasificacion.falso import ClasificadorFalso
from cobro_vision.clasificacion.instancias import ConstructorInstancias
from cobro_vision.clasificacion.svm import ClasificadorSVM
from cobro_vision.configuracion import Configuracion
from cobro_vision.controlador.controlador import ControladorTransaccion
from cobro_vision.dominio.errores import ConfiguracionInvalida
from cobro_vision.explicacion.trazabilidad import Trazabilidad
from cobro_vision.negocio.calculadora import CalculadoraCosto
from cobro_vision.negocio.catalogo import CatalogoAlimentos
from cobro_vision.negocio.politicas import CobroPorComponente, CobroPorPlato
from cobro_vision.negocio.posprocesador import Posprocesador
from cobro_vision.preprocesamiento.detector_plato import DetectorPlato
from cobro_vision.preprocesamiento.preprocesador import Preprocesador
from cobro_vision.segmentacion.base import Segmentador
from cobro_vision.segmentacion.superpixeles import SegmentadorSuperpixeles
from cobro_vision.segmentacion.ventanas import SegmentadorVentanas


class FabricaPipeline:
    """Ensambla operación y entrenamiento con los mismos componentes de visión."""

    _camaras: ClassVar[dict[str, type]] = {
        "CamaraArchivo": CamaraArchivo,
        "CamaraUSB": CamaraUSB,
    }
    _segmentadores: ClassVar[dict[str, type]] = {
        "SegmentadorSuperpixeles": SegmentadorSuperpixeles,
        "SegmentadorVentanas": SegmentadorVentanas,
    }
    _extractores: ClassVar[dict[str, type]] = {
        "ExtractorHistogramaColor": ExtractorHistogramaColor,
        "ExtractorTextura": ExtractorTextura,
    }
    _clasificadores: ClassVar[dict[str, type]] = {
        "falso": ClasificadorFalso,
        "svm": ClasificadorSVM,
    }
    _politicas: ClassVar[dict[str, type]] = {
        "CobroPorComponente": CobroPorComponente,
        "CobroPorPlato": CobroPorPlato,
    }

    @staticmethod
    def crear(config: Configuracion) -> ControladorTransaccion:
        """Ensambla las variantes registradas sin entrenar ningún clasificador."""
        camara_config = config.seccion("camara")
        clasificacion = config.seccion("clasificador")
        cobro = config.seccion("cobro")
        camara_clase = FabricaPipeline._elegir(
            FabricaPipeline._camaras, camara_config["clase"], "camara.clase"
        )
        preprocesador, detector, segmentador, constructor = (
            FabricaPipeline._crear_vision(config)
        )
        clasificador_clase = FabricaPipeline._elegir(
            FabricaPipeline._clasificadores, clasificacion["tipo"], "clasificador.tipo"
        )
        politica_clase = FabricaPipeline._elegir(
            FabricaPipeline._politicas, cobro["politica"], "cobro.politica"
        )
        rutas = config.seccion("rutas")
        catalogo = CatalogoAlimentos(Path(rutas["catalogo"]))
        # TODO(equipo): si aún no existe el catálogo se inicializa con la semilla
        # de ejemplo para permitir demo; uno existente conserva sus precios.
        if not catalogo.ruta_db.exists():
            catalogo.inicializar(Path(rutas["semilla_catalogo"]))
        if clasificacion["tipo"] == "svm":
            ruta_modelo = Path(rutas["modelo_actual"])
            if not ruta_modelo.is_file():
                raise FileNotFoundError(
                    f"No existe el modelo SVM: {ruta_modelo}. "
                    "Ejecute python -m cobro_vision entrenar."
                )
            clasificador = clasificador_clase.cargar(
                ruta_modelo, constructor, clasificacion["umbral_confianza"]
            )
        else:
            clasificador = clasificador_clase(catalogo.clases_validas(), config.semilla)
            clasificador.nombres_caracteristicas = constructor.nombres()
        politica = (
            politica_clase(**cobro["por_plato"])
            if cobro["politica"] == "CobroPorPlato"
            else politica_clase()
        )
        camara = (
            camara_clase(Path(camara_config["carpeta"]))
            if camara_config["clase"] == "CamaraArchivo"
            else camara_clase(indice=camara_config["indice"])
        )
        return ControladorTransaccion(
            config=config,
            camara=camara,
            preprocesador=preprocesador,
            detector=detector,
            segmentador=segmentador,
            constructor=constructor,
            clasificador=clasificador,
            posprocesador=Posprocesador(
                config.seccion("posproceso"),
                catalogo,
                clasificacion["umbral_confianza"],
            ),
            calculadora=CalculadoraCosto(catalogo, politica),
            trazabilidad=Trazabilidad(),
        )

    @staticmethod
    def crear_entrenador(config: Configuracion) -> EntrenadorModelo:
        """Prepara entrenamiento sin cargar un modelo anterior ni abrir la cámara."""
        preprocesador, detector, segmentador, constructor = (
            FabricaPipeline._crear_vision(config)
        )
        catalogo = CatalogoAlimentos(Path(config.seccion("rutas")["catalogo"]))
        instancias = ConstructorInstancias(
            config,
            preprocesador,
            detector,
            segmentador,
            constructor,
            MapaEtiquetas(catalogo.clases_validas()),
        )
        return EntrenadorModelo(config, instancias, constructor)

    @staticmethod
    def _crear_vision(
        config: Configuracion,
    ) -> tuple[Preprocesador, DetectorPlato, Segmentador, ConstructorVector]:
        segmentacion = config.seccion("segmentacion")
        segmentador_clase = FabricaPipeline._elegir(
            FabricaPipeline._segmentadores, segmentacion["clase"], "segmentacion.clase"
        )
        extractores = [
            FabricaPipeline._elegir(
                FabricaPipeline._extractores,
                extractor["clase"],
                "caracteristicas.clase",
            )(extractor)
            for extractor in config.seccion("caracteristicas")["extractores"]
        ]
        return (
            Preprocesador(config.seccion("imagen")),
            DetectorPlato(config.seccion("plato")),
            segmentador_clase(segmentacion),
            ConstructorVector(extractores),
        )

    @staticmethod
    def _crear_svm_entrenado(
        pipeline: object,
        clases: list[str],
        umbral: float,
        meta: dict,
    ) -> ClasificadorSVM:
        return FabricaPipeline._clasificadores["svm"](pipeline, clases, umbral, meta)

    @staticmethod
    def _elegir(registro: dict, nombre: str, campo: str) -> type:
        try:
            return registro[nombre]
        except KeyError as error:
            raise ConfiguracionInvalida(
                f"{campo}: componente desconocido: {nombre}"
            ) from error
