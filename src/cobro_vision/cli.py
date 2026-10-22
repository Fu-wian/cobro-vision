"""Interfaz de línea de comandos del prototipo."""

import argparse
import logging
import sqlite3
from pathlib import Path

from cobro_vision.configuracion import Configuracion
from cobro_vision.dominio.errores import CobroVisionError
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
