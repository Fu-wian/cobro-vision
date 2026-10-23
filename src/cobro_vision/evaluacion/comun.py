"""Lectura estricta de test y exportación local de los informes."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from cobro_vision.adquisicion.gestor_dataset import GestorDataset
from cobro_vision.configuracion import Configuracion
from cobro_vision.dominio.errores import EtiquetaInvalida
from cobro_vision.dominio.modelos import Muestra
from cobro_vision.evaluacion.registro import RegistroExperimentos
from cobro_vision.negocio.catalogo import CatalogoAlimentos


def cargar_prueba(
    config: Configuracion,
) -> tuple[CatalogoAlimentos, GestorDataset, list[Muestra]]:
    """Exige un manifiesto dividido; nunca lo construye ni lee etiquetas de train."""
    rutas = config.seccion("rutas")
    if not Path(rutas["manifiesto"]).is_file():
        raise FileNotFoundError(
            "Falta el manifiesto; ejecute construir-manifiesto y dividir antes de evaluar."
        )
    catalogo = CatalogoAlimentos(Path(rutas["catalogo"]))
    catalogo.clases_validas()
    gestor = GestorDataset(config, catalogo)
    muestras = gestor.cargar_muestras("test")
    if not muestras:
        raise EtiquetaInvalida(
            "No hay muestras test etiquetadas; revise el split con dividir."
        )
    return catalogo, gestor, muestras


def guardar_evaluacion(
    nombre: str,
    config: Configuracion,
    metricas: dict,
    tablas: dict[str, pd.DataFrame],
    registro: RegistroExperimentos,
    matrices: dict[str, pd.DataFrame] | None = None,
) -> Path:
    """Escribe JSON, CSV y matrices PNG sin depender de una pantalla."""
    metricas["clasificador"] = config.seccion("clasificador")["tipo"]
    carpeta = (
        Path(config.seccion("rutas")["salidas"])
        / "evaluacion"
        / datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    )
    carpeta.mkdir(parents=True, exist_ok=False)
    (carpeta / "metricas.json").write_text(
        json.dumps(
            {
                "nombre": nombre,
                "split": "test",
                "hash_configuracion": config.hash(),
                "metricas": metricas,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    for archivo, tabla in tablas.items():
        tabla.to_csv(carpeta / f"{archivo}.csv", index=False, encoding="utf-8")
    for archivo, matriz in (matrices or {}).items():
        matriz.to_csv(carpeta / f"{archivo}.csv", index_label="real", encoding="utf-8")
        figura = Figure(figsize=(8, 7), layout="constrained")
        lienzo = FigureCanvasAgg(figura)
        eje = figura.subplots()
        imagen = eje.imshow(matriz.to_numpy(), cmap="Blues", vmin=0)
        eje.set(
            xticks=range(len(matriz.columns)),
            yticks=range(len(matriz.index)),
            xticklabels=matriz.columns,
            yticklabels=matriz.index,
            xlabel="Clase predicha",
            ylabel="Clase real",
            title="Matriz de confusión (test)",
        )
        eje.tick_params(axis="x", labelrotation=90)
        for fila in range(len(matriz)):
            for columna in range(len(matriz.columns)):
                eje.text(
                    columna,
                    fila,
                    str(matriz.iat[fila, columna]),
                    ha="center",
                    va="center",
                )
        figura.colorbar(imagen, ax=eje)
        lienzo.print_png(str(carpeta / f"{archivo}.png"))
    registro.registrar(nombre, config, metricas)
    return carpeta
