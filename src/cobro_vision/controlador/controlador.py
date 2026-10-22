"""Orquestación de una fotografía hasta su recibo, sin interfaz gráfica."""

from dataclasses import dataclass, field
from datetime import datetime
from itertools import count

import numpy as np

from cobro_vision.adquisicion.camara import Camara
from cobro_vision.caracteristicas.constructor import ConstructorVector
from cobro_vision.clasificacion.base import Clasificador
from cobro_vision.configuracion import Configuracion
from cobro_vision.controlador.medidor import MedidorTiempos
from cobro_vision.dominio.errores import PlatoNoDetectado
from cobro_vision.dominio.modelos import Recibo, Transaccion
from cobro_vision.explicacion.trazabilidad import Trazabilidad
from cobro_vision.negocio.calculadora import CalculadoraCosto
from cobro_vision.negocio.posprocesador import Posprocesador
from cobro_vision.preprocesamiento.detector_plato import DetectorPlato
from cobro_vision.preprocesamiento.preprocesador import Preprocesador
from cobro_vision.segmentacion.base import Segmentador

_SECUENCIA = count()


@dataclass
class ControladorTransaccion:
    """Ejecuta las siete etapas con las dependencias elegidas por la fábrica."""

    config: Configuracion
    camara: Camara
    preprocesador: Preprocesador
    detector: DetectorPlato
    segmentador: Segmentador
    constructor: ConstructorVector
    clasificador: Clasificador
    posprocesador: Posprocesador
    calculadora: CalculadoraCosto
    trazabilidad: Trazabilidad
    medidor: MedidorTiempos = field(default_factory=MedidorTiempos)

    def procesar(self, imagen: np.ndarray) -> Transaccion:
        """Procesa la foto y devuelve también los fallos de detección del plato."""
        self.medidor.reiniciar()
        identificador = (
            f"T-{datetime.now():%Y%m%d-%H%M%S}-{next(_SECUENCIA) % 65536:04x}"
        )
        transaccion = Transaccion(identificador, imagen.copy())
        try:
            with self.medidor.medir("preprocesamiento"):
                transaccion.imagen_procesada = self.preprocesador.procesar(imagen)
            self._registrar(
                transaccion,
                "preprocesamiento",
                "Imagen preparada en BGR.",
                {
                    "forma": list(transaccion.imagen_procesada.shape),
                    "parametros": self.config.seccion("imagen"),
                },
            )
            with self.medidor.medir("deteccion_plato"):
                transaccion.mascara_plato = self.detector.detectar(
                    transaccion.imagen_procesada
                )
            self._registrar(
                transaccion,
                "deteccion_plato",
                "Plato detectado.",
                {
                    "area": int(transaccion.mascara_plato.sum()),
                    "parametros": self.config.seccion("plato"),
                },
            )
            with self.medidor.medir("segmentacion"):
                transaccion.regiones_crudas = self.segmentador.segmentar(
                    transaccion.imagen_procesada,
                    transaccion.mascara_plato,
                )
            self._registrar(
                transaccion,
                "segmentacion",
                "Regiones segmentadas.",
                {
                    "metodo": self.config.seccion("segmentacion")["clase"],
                    "cantidad": len(transaccion.regiones_crudas),
                    "parametros": self.config.seccion("segmentacion"),
                },
            )
            with self.medidor.medir("caracteristicas"):
                # construir_lote crea un único ContextoImagen para todas las regiones.
                vectores = self.constructor.construir_lote(
                    transaccion.imagen_procesada,
                    transaccion.regiones_crudas,
                )
            self._registrar(
                transaccion,
                "caracteristicas",
                "Vectores construidos.",
                {
                    "dimension": self.constructor.dimension(),
                    "cantidad": len(vectores),
                },
            )
            with self.medidor.medir("clasificacion"):
                resultados = self.clasificador.predecir_lote(vectores)
                if len(resultados) != len(transaccion.regiones_crudas):
                    raise ValueError(
                        "El clasificador debe devolver un resultado por región."
                    )
                for region, resultado in zip(transaccion.regiones_crudas, resultados):
                    region.resultado = resultado
            self._registrar(
                transaccion,
                "clasificacion",
                "Regiones clasificadas.",
                {
                    "clasificador": type(self.clasificador).__name__,
                    "cantidad": len(resultados),
                    "umbral": self.config.seccion("clasificador")["umbral_confianza"],
                },
            )
            with self.medidor.medir("posprocesamiento"):
                transaccion.regiones = self.posprocesador.consolidar(
                    transaccion.regiones_crudas,
                    transaccion.mascara_plato,
                )
            for evento in self.posprocesador.eventos:
                self.trazabilidad.registrar(
                    transaccion,
                    "posprocesamiento",
                    "Regla de consolidación aplicada.",
                    evento,
                )
            self._registrar(
                transaccion,
                "posprocesamiento",
                "Regiones consolidadas.",
                {
                    "cantidad": len(transaccion.regiones),
                    "parametros": self.config.seccion("posproceso"),
                    "regiones": [
                        {
                            "id": r.id,
                            "clase": r.resultado.id_clase,
                            "area": r.area,
                            "score": r.resultado.score,
                            "estado": r.estado,
                            "ids_origen": self.posprocesador.origenes[r.id],
                        }
                        for r in transaccion.regiones
                    ],
                },
            )
            with self.medidor.medir("cobro"):
                transaccion.recibo = self.calculadora.calcular(transaccion.regiones)
                # El controlador sí conoce el área real del plato (sin cambiar calcular).
                area_plato = int(transaccion.mascara_plato.sum())
                for item in transaccion.recibo.items:
                    item.area_frac = (
                        sum(
                            r.area
                            for r in transaccion.regiones
                            if r.id in item.ids_region
                        )
                        / area_plato
                        if area_plato
                        else 0.0
                    )
            self._registrar(
                transaccion,
                "cobro",
                "Recibo calculado en centavos.",
                {
                    "politica": self.config.seccion("cobro")["politica"],
                    "cantidad": len(transaccion.recibo.items),
                    "total_centavos": transaccion.recibo.total_centavos,
                },
            )
            transaccion.estado = (
                "requiere_revision"
                if transaccion.recibo.requiere_revision
                else "procesada"
            )
        except PlatoNoDetectado as error:
            transaccion.estado = "error"
            transaccion.mensaje_error = str(error)
            transaccion.recibo = Recibo([])
            self._registrar(
                transaccion,
                "deteccion_plato",
                "No se pudo detectar el plato.",
                {
                    "error": str(error),
                },
            )
        finally:
            transaccion.tiempos = self.medidor.resumen()
        return transaccion

    def capturar_y_procesar(self) -> Transaccion:
        """Obtiene una foto del origen configurado y ejecuta el pipeline."""
        return self.procesar(self.camara.capturar())

    def _registrar(
        self, transaccion: Transaccion, etapa: str, descripcion: str, datos: dict
    ) -> None:
        self.trazabilidad.registrar(
            transaccion,
            etapa,
            descripcion,
            datos,
            self.medidor.resumen().get(etapa),
        )
