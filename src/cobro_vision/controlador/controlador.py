"""Orquestación de una fotografía hasta su recibo, sin interfaz gráfica."""

import csv
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from itertools import count
from pathlib import Path

import numpy as np

from cobro_vision.adquisicion.camara import Camara, CamaraArchivo
from cobro_vision.caracteristicas.constructor import ConstructorVector
from cobro_vision.clasificacion.base import Clasificador
from cobro_vision.configuracion import Configuracion
from cobro_vision.controlador.medidor import MedidorTiempos
from cobro_vision.dominio.errores import PlatoNoDetectado
from cobro_vision.dominio.modelos import (
    ID_FONDO,
    Ajuste,
    ItemRecibo,
    Recibo,
    Region,
    Transaccion,
)
from cobro_vision.explicacion.trazabilidad import Trazabilidad
from cobro_vision.negocio.calculadora import CalculadoraCosto
from cobro_vision.negocio.catalogo import CatalogoAlimentos
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

    @property
    def catalogo(self) -> CatalogoAlimentos:
        """Devuelve el catálogo del pipeline para consultas de presentación."""
        return self.calculadora.catalogo

    def cargar_y_procesar(self, ruta: Path) -> Transaccion:
        """Lee la foto elegida y la procesa con las dependencias configuradas."""
        return self.procesar(CamaraArchivo.leer(ruta))

    def corregir_region(
        self, t: Transaccion, id_region: int, id_clase: str
    ) -> Transaccion:
        """Corrige una región activa y recalcula el recibo."""
        self._validar_edicion(t)
        self.catalogo.obtener(id_clase)
        region = next((r for r in t.regiones if r.id == id_region), None)
        if region is None or region.resultado is None or region.estado == "eliminada":
            raise ValueError(f"No existe una región activa con id {id_region}.")
        clase_anterior = region.resultado.id_clase
        self._corregir(t, region, id_clase)
        self._cerrar_alta_anterior(t, clase_anterior, id_clase)
        return self._recalcular(t)

    def corregir_item(
        self, t: Transaccion, id_clase_actual: str, id_clase_nueva: str
    ) -> Transaccion:
        """Corrige todas las regiones del ítem con un único recálculo."""
        self._validar_edicion(t)
        self._validar_cobrable(id_clase_nueva)
        item = self._buscar_item(t, id_clase_actual)
        if item.origen == "manual":
            t.ajustes.extend(
                [Ajuste("quitar", id_clase_actual), Ajuste("agregar", id_clase_nueva)]
            )
        else:
            for region in t.regiones:
                if region.id in item.ids_region:
                    self._corregir(t, region, id_clase_nueva)
            self._cerrar_alta_anterior(t, id_clase_actual, id_clase_nueva)
        return self._recalcular(t)

    def quitar_item(self, t: Transaccion, id_clase: str) -> Transaccion:
        """Elimina el ítem y oculta sus regiones conservando la auditoría."""
        self._validar_edicion(t)
        item = self._buscar_item(t, id_clase)
        for region in t.regiones:
            if region.id in item.ids_region:
                region.estado = "eliminada"
        t.ajustes.append(Ajuste("quitar", id_clase))
        return self._recalcular(t)

    def agregar_item(self, t: Transaccion, id_clase: str) -> Transaccion:
        """Añade una clase manual sin duplicar alimentos ya presentes."""
        self._validar_edicion(t)
        self._validar_cobrable(id_clase)
        t.ajustes.append(Ajuste("agregar", id_clase))
        return self._recalcular(t)

    def confirmar(self, t: Transaccion) -> Transaccion:
        """Guarda una fila CSV antes de dar por confirmada la transacción."""
        # TODO(equipo): confirmar dos veces es idempotente; no se cobra un recibo vacío.
        if t.estado == "confirmada":
            return t
        self._validar_edicion(t)
        if not t.recibo.items:
            raise ValueError("Agregue al menos un ítem antes de confirmar.")
        if self.config.seccion("ui")["exigir_revision"] and t.recibo.requiere_revision:
            raise ValueError("Revise los ítems marcados antes de confirmar.")
        ruta = Path(self.config.seccion("rutas")["salidas"]) / "transacciones.csv"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        # TODO(equipo): columnas CSV explícitas; ítems y tiempos se guardan como JSON,
        # y el total como centavos enteros para conservar el importe exacto.
        fila = {
            "id": t.id,
            "fecha_hora": datetime.now().astimezone().isoformat(timespec="seconds"),
            "items": json.dumps(
                [asdict(i) for i in t.recibo.items], ensure_ascii=False
            ),
            "total_centavos": t.recibo.total_centavos,
            "tiempos": json.dumps(t.tiempos, ensure_ascii=False, sort_keys=True),
            "n_ajustes": len(t.ajustes),
        }
        with ruta.open("a", encoding="utf-8", newline="") as archivo:
            escritor = csv.DictWriter(archivo, fieldnames=list(fila))
            if archivo.tell() == 0:
                escritor.writeheader()
            escritor.writerow(fila)
        t.estado = "confirmada"
        return t

    @staticmethod
    def _validar_edicion(t: Transaccion) -> None:
        if t.estado not in {"procesada", "requiere_revision"} or t.recibo is None:
            raise ValueError("La transacción no está disponible para modificaciones.")

    def _validar_cobrable(self, id_clase: str) -> None:
        alimento = self.catalogo.obtener(id_clase)
        if alimento.id_clase == ID_FONDO or alimento.categoria == "fondo":
            raise ValueError("El fondo no es un alimento cobrable.")

    @staticmethod
    def _buscar_item(t: Transaccion, id_clase: str) -> ItemRecibo:
        item = next(
            (i for i in t.recibo.items if i.alimento.id_clase == id_clase), None
        )
        if item is None:
            raise ValueError(f"No existe el ítem {id_clase} en el recibo.")
        return item

    @staticmethod
    def _corregir(t: Transaccion, region: Region, id_clase: str) -> None:
        region.resultado = replace(
            region.resultado,
            id_clase=id_clase,
            requiere_revision=False,
            alternativa=None,
        )
        region.estado = "corregida"
        t.ajustes.append(Ajuste("corregir", id_clase, region.id))

    def _recalcular(self, t: Transaccion) -> Transaccion:
        t.recibo = self.calculadora.calcular(t.regiones, t.ajustes)
        area_plato = int(t.mascara_plato.sum()) if t.mascara_plato is not None else 0
        for item in t.recibo.items:
            if item.origen == "auto" and area_plato:
                item.area_frac = (
                    sum(r.area for r in t.regiones if r.id in item.ids_region)
                    / area_plato
                )
        t.estado = "requiere_revision" if t.recibo.requiere_revision else "procesada"
        self.trazabilidad.registrar(
            t,
            "ajuste",
            "Recibo recalculado tras la revisión del cajero.",
            {"n_ajustes": len(t.ajustes), "total_centavos": t.recibo.total_centavos},
        )
        return t

    @staticmethod
    def _cerrar_alta_anterior(t: Transaccion, anterior: str, nueva: str) -> None:
        # TODO(equipo): si se mueve la última región de una clase agregada antes,
        # registrar su baja evita que el alta histórica reaparezca en el recibo.
        if (
            anterior != nueva
            and any(a.tipo == "agregar" and a.id_clase == anterior for a in t.ajustes)
            and not any(
                r.resultado
                and r.resultado.id_clase == anterior
                and r.estado != "eliminada"
                for r in t.regiones
            )
        ):
            t.ajustes.append(Ajuste("quitar", anterior))

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
