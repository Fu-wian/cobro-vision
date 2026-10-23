"""Interfaz de línea de comandos del prototipo."""

import argparse
import logging
import sqlite3
from pathlib import Path

import cv2

from cobro_vision.adquisicion.camara import CamaraArchivo
from cobro_vision.adquisicion.gestor_dataset import GestorDataset
from cobro_vision.adquisicion.resumen_dataset import resumir_dataset
from cobro_vision.adquisicion.validador_etiquetas import ValidadorEtiquetas
from cobro_vision.configuracion import Configuracion
from cobro_vision.controlador.fabrica import FabricaPipeline
from cobro_vision.dominio.errores import CobroVisionError
from cobro_vision.dominio.modelos import Transaccion
from cobro_vision.dominio.utilidades import formatear_dinero
from cobro_vision.explicacion.depuracion import guardar_depuracion
from cobro_vision.negocio.catalogo import CatalogoAlimentos


def construir_analizador() -> argparse.ArgumentParser:
    """Define los comandos y argumentos previstos en la especificación."""
    analizador = argparse.ArgumentParser(
        prog="python -m cobro_vision",
        description="Identificación de alimentos y cobro con visión artificial clásica.",
    )
    analizador.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Ruta de configuración (por defecto: config.yaml).",
    )
    subcomandos = analizador.add_subparsers(dest="comando", required=True)
    especificaciones = (
        ("inicializar-catalogo", "Crear el catálogo desde la semilla.", 1),
        ("validar-etiquetas", "Validar las etiquetas de LabelMe.", 4),
        ("construir-manifiesto", "Crear o actualizar el manifiesto del dataset.", 4),
        ("dividir", "Dividir el dataset por plato entre entrenamiento y prueba.", 4),
        ("resumen-dataset", "Mostrar la cobertura del dataset por clase.", 4),
        ("demo", "Procesar una foto y mostrar su recibo.", 2),
        ("entrenar", "Entrenar y guardar el clasificador SVM.", 5),
        ("evaluar", "Evaluar el clasificador, la segmentación o el sistema.", 6),
        ("ablacion", "Comparar variantes y registrar los experimentos.", 7),
        ("app", "Abrir la interfaz de escritorio del cajero.", 3),
    )
    for nombre, descripcion, etapa in especificaciones:
        subanalizador = subcomandos.add_parser(
            nombre, help=descripcion, description=descripcion
        )
        subanalizador.add_argument(
            "--config",
            type=Path,
            default=argparse.SUPPRESS,
            help="Ruta de configuración (por defecto: config.yaml).",
        )
        subanalizador.set_defaults(etapa=etapa)
        if nombre == "dividir":
            subanalizador.add_argument(
                "--prueba",
                type=float,
                required=True,
                help="Fracción de prueba, p. ej. 0.2.",
            )
        elif nombre == "resumen-dataset":
            subanalizador.add_argument(
                "--csv", type=Path, help="Guardar también la tabla en esta ruta CSV."
            )
        elif nombre == "demo":
            subanalizador.add_argument(
                "--imagen", type=Path, required=True, help="Ruta de la foto a procesar."
            )
            subanalizador.add_argument(
                "--debug", action="store_true", help="Guardar imágenes de depuración."
            )
        elif nombre == "entrenar":
            subanalizador.add_argument(
                "--modo",
                choices=("poligonos", "segmentacion", "ambos"),
                help="Modo de generación de instancias (por defecto: config.yaml).",
            )
        elif nombre == "evaluar":
            subanalizador.add_argument(
                "objetivo",
                choices=("clasificador", "segmentacion", "sistema"),
                help="Componente que se evaluará.",
            )
    return analizador


def principal(argumentos: list[str] | None = None) -> int:
    """Ejecuta los comandos disponibles o informa la etapa pendiente."""
    logging.basicConfig(level=logging.INFO)
    opciones = construir_analizador().parse_args(argumentos)
    if opciones.comando == "demo":
        return _ejecutar_demo(opciones)
    if opciones.comando == "app":
        return _ejecutar_app(opciones)
    if opciones.comando == "entrenar":
        return _ejecutar_entrenamiento(opciones)
    if opciones.comando in (
        "validar-etiquetas",
        "construir-manifiesto",
        "dividir",
        "resumen-dataset",
    ):
        return _ejecutar_dataset(opciones)
    if opciones.comando == "inicializar-catalogo":
        try:
            config = Configuracion.cargar(opciones.config)
            rutas = config.seccion("rutas")
            ruta_catalogo = Path(rutas["catalogo"])
            catalogo = CatalogoAlimentos(ruta_catalogo)
            catalogo.inicializar(Path(rutas["semilla_catalogo"]))
            # TODO(equipo): No hay ruta de clases en el esquema; clases.txt va
            # junto al catálogo (data/clases.txt con la configuración por defecto).
            ruta_clases = ruta_catalogo.with_name("clases.txt")
            ruta_clases.write_text(
                "".join(f"{clase}\n" for clase in catalogo.clases_validas()),
                encoding="utf-8",
            )
        except (CobroVisionError, OSError, ValueError, sqlite3.Error) as error:
            logging.getLogger(__name__).error(
                "No se pudo inicializar el catálogo: %s", error
            )
            return 1
        print(f"Catálogo inicializado: {ruta_catalogo}; etiquetas: {ruta_clases}")
        return 0
    print(f"No implementado aún (etapa {opciones.etapa})")
    return 2


def _ejecutar_entrenamiento(opciones: argparse.Namespace) -> int:
    try:
        config = Configuracion.cargar(opciones.config)
        if opciones.modo is not None:
            datos = config._datos.copy()
            datos["entrenamiento"] = {
                **config.seccion("entrenamiento"),
                "modo": opciones.modo,
            }
            config = Configuracion(datos)
        catalogo = CatalogoAlimentos(Path(config.seccion("rutas")["catalogo"]))
        gestor = GestorDataset(config, catalogo)
        muestras = gestor.cargar_muestras()
        entrenador = FabricaPipeline.crear_entrenador(config)
        modelo = entrenador.entrenar(muestras)
        print(
            f"SVM entrenado ({modelo.meta['modo']}): "
            f"F1 macro CV = {modelo.meta['metricas_cv']['f1_macro_media']:.4f}. "
            f"Modelo actual: {config.seccion('rutas')['modelo_actual']}"
        )
        return 0
    except (CobroVisionError, OSError, ValueError, sqlite3.Error, cv2.error) as error:
        logging.getLogger(__name__).error("No se pudo entrenar el SVM: %s", error)
        return 1


def _ejecutar_dataset(opciones: argparse.Namespace) -> int:
    try:
        config = Configuracion.cargar(opciones.config)
        rutas = config.seccion("rutas")
        catalogo = CatalogoAlimentos(Path(rutas["catalogo"]))
        # No hacer upsert aquí: el catálogo puede contener cambios del equipo.
        catalogo.clases_validas()
        gestor = GestorDataset(config, catalogo)
        if opciones.comando == "validar-etiquetas":
            informe = ValidadorEtiquetas(config, catalogo).validar([])
            for mensaje in informe.errores:
                print(f"ERROR: {mensaje}")
            for mensaje in informe.advertencias:
                print(f"ADVERTENCIA: {mensaje}")
            print(
                f"Validación: {len(informe.errores)} errores, {len(informe.advertencias)} advertencias."
            )
            return 0 if informe.ok else 1
        if opciones.comando == "resumen-dataset":
            tabla = resumir_dataset(config, catalogo)
            if opciones.csv is not None:
                opciones.csv.parent.mkdir(parents=True, exist_ok=True)
                tabla.to_csv(
                    opciones.csv, index=False, encoding="utf-8", lineterminator="\n"
                )
            print(tabla.to_string(index=False))
        elif opciones.comando == "construir-manifiesto":
            manifiesto = gestor.construir_manifiesto()
            print(
                f"Manifiesto guardado en {rutas['manifiesto']}: {len(manifiesto)} fotos."
            )
        else:
            gestor.dividir(opciones.prueba, config.semilla)
            print(f"División por plato_id guardada en {rutas['manifiesto']}.")
        return 0
    except (CobroVisionError, OSError, ValueError, sqlite3.Error, cv2.error) as error:
        logging.getLogger(__name__).error(
            "No se pudo ejecutar %s: %s. Si falta el catálogo, ejecute inicializar-catalogo.",
            opciones.comando,
            error,
        )
        return 1


def _ejecutar_app(opciones: argparse.Namespace) -> int:
    # Importación diferida: los comandos de consola funcionan sin Tk ni pantalla.
    controlador = None
    vista = None
    try:
        from cobro_vision.vista.vista_cajero import VistaCajero

        config = Configuracion.cargar(opciones.config)
        controlador = FabricaPipeline.crear(config)
        vista = VistaCajero(controlador, config)
        vista.mainloop()
        return 0
    except Exception as error:  # noqa: BLE001 - incluye errores de Tk y de arranque
        logging.getLogger(__name__).error("No se pudo abrir la interfaz: %s", error)
        return 1
    finally:
        if vista is not None:
            vista.destroy()
        elif controlador is not None:
            controlador.camara.liberar()


def _ejecutar_demo(opciones: argparse.Namespace) -> int:
    controlador = None
    try:
        config = Configuracion.cargar(opciones.config)
        imagen = CamaraArchivo.leer(opciones.imagen)
        controlador = FabricaPipeline.crear(config)
        transaccion = controlador.procesar(imagen)
        texto = _texto_recibo(transaccion, config.seccion("ui")["moneda"])
        print(texto)
        if opciones.debug:
            carpeta = (
                Path(config.seccion("rutas")["salidas"])
                / "debug"
                / opciones.imagen.stem
            )
            guardar_depuracion(transaccion, carpeta, texto)
            logging.getLogger(__name__).info("Depuración guardada en %s", carpeta)
        if transaccion.estado == "error":
            logging.getLogger(__name__).error("%s", transaccion.mensaje_error)
            return 1
        return 0
    except (
        CobroVisionError,
        OSError,
        ValueError,
        RuntimeError,
        sqlite3.Error,
        cv2.error,
    ) as error:
        logging.getLogger(__name__).error("No se pudo procesar la imagen: %s", error)
        return 1
    finally:
        if controlador is not None:
            controlador.camara.liberar()


def _texto_recibo(transaccion: Transaccion, moneda: str) -> str:
    lineas = [f"Recibo {transaccion.id}"]
    if transaccion.recibo is not None:
        for item in transaccion.recibo.items:
            precio = formatear_dinero(item.precio_cobrado_centavos, moneda)
            marca = " [REVISAR]" if item.requiere_revision else ""
            lineas.append(f"{item.alimento.nombre}: {precio}{marca}")
        lineas.append(
            f"Total: {formatear_dinero(transaccion.recibo.total_centavos, moneda)}"
        )
    lineas.extend(
        [
            f"Estado: {transaccion.estado}",
            f"Tiempo total: {transaccion.tiempos.get('total', 0):.2f} ms",
        ]
    )
    return "\n".join(lineas)
