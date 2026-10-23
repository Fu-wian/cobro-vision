"""Perfila cinco fotos reales y compara la salida exacta entre implementaciones."""

import argparse
import cProfile
import hashlib
import importlib.util
import json
import platform
import pstats
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter

from cobro_vision.configuracion import Configuracion
from cobro_vision.controlador.fabrica import FabricaPipeline
from cobro_vision.dominio.modelos import Region
from cobro_vision.negocio.posprocesador import Posprocesador


def _resumir(regiones: list[Region], posprocesador: Posprocesador) -> dict:
    return {
        "regiones": [
            {
                "id": region.id,
                "mascara_sha256": hashlib.sha256(region.mascara.tobytes()).hexdigest(),
                "bbox": list(region.bbox),
                "area": region.area,
                "vector": None if region.vector is None else region.vector.tolist(),
                "resultado": {
                    **asdict(region.resultado),
                    "score": region.resultado.score.hex(),
                },
                "estado": region.estado,
            }
            for region in regiones
        ],
        "eventos": posprocesador.eventos.copy(),
        "origenes": {str(k): v for k, v in posprocesador.origenes.items()},
    }


def principal() -> None:
    """Mide consolidar por separado y conserva perfiles y huellas de sus salidas."""
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("--config", type=Path, default=Path("config.yaml"))
    analizador.add_argument("--salida", type=Path, required=True)
    analizador.add_argument("--implementacion", type=Path)
    analizador.add_argument("--comparar", type=Path)
    argumentos = analizador.parse_args()
    config = Configuracion.cargar(argumentos.config)
    segmentacion = config.seccion("segmentacion")
    if (segmentacion["clase"], segmentacion["n_segmentos"]) != (
        "SegmentadorSuperpixeles",
        100,
    ):
        analizador.error(
            "El perfil requiere SegmentadorSuperpixeles con n_segmentos: 100."
        )
    fotos = sorted(
        ruta
        for ruta in Path(config.seccion("rutas")["imagenes"]).rglob("*")
        if ruta.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:5]
    if len(fotos) < 3:
        analizador.error("Se requieren al menos tres fotos en la carpeta configurada.")
    controlador = FabricaPipeline.crear_evaluacion(config)
    if argumentos.implementacion:
        especificacion = importlib.util.spec_from_file_location(
            "posprocesador_referencia", argumentos.implementacion
        )
        modulo = importlib.util.module_from_spec(especificacion)
        especificacion.loader.exec_module(modulo)
        controlador.posprocesador = modulo.Posprocesador(
            config.seccion("posproceso"),
            controlador.posprocesador._catalogo,
            config.seccion("clasificador")["umbral_confianza"],
        )
    referencia = (
        json.loads(argumentos.comparar.read_text(encoding="utf-8"))["fotos"]
        if argumentos.comparar
        else None
    )
    argumentos.salida.mkdir(parents=True, exist_ok=True)
    perfil = cProfile.Profile()
    medidas = []
    for indice, foto in enumerate(fotos):
        transaccion = controlador.procesar(controlador.camara.leer(foto))
        if transaccion.estado == "error":
            raise RuntimeError(f"{foto}: {transaccion.mensaje_error}")
        posprocesador = controlador.posprocesador
        salida = _resumir(transaccion.regiones, posprocesador)
        tiempos = []
        for _ in range(3):
            inicio = perf_counter()
            regiones = posprocesador.consolidar(
                transaccion.regiones_crudas, transaccion.mascara_plato
            )
            tiempos.append(1000 * (perf_counter() - inicio))
            assert _resumir(regiones, posprocesador) == salida
        regiones = perfil.runcall(
            posprocesador.consolidar,
            transaccion.regiones_crudas,
            transaccion.mascara_plato,
        )
        assert _resumir(regiones, posprocesador) == salida
        if referencia is not None:
            assert referencia[indice]["foto"] == foto.as_posix()
            assert referencia[indice]["salida"] == salida, foto
        medida = {
            "foto": foto.as_posix(),
            "forma": list(transaccion.mascara_plato.shape),
            "regiones_crudas": len(transaccion.regiones_crudas),
            "tiempos_consolidar_ms": tiempos,
            "mediana_consolidar_ms": median(tiempos),
            "pipeline_ms": transaccion.tiempos,
            "salida": salida,
        }
        medidas.append(medida)
        print(f"{foto}: consolidar={median(tiempos):.3f} ms")
    perfil.dump_stats(str(argumentos.salida / "perfil.prof"))
    with (argumentos.salida / "perfil.txt").open("w", encoding="utf-8") as archivo:
        pstats.Stats(perfil, stream=archivo).strip_dirs().sort_stats(
            "cumtime"
        ).print_stats(30)
    (argumentos.salida / "resumen.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "config_hash": config.hash(),
                "fotos": medidas,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if referencia is not None:
        print("Salidas idénticas a la referencia en todas las fotos y repeticiones.")


if __name__ == "__main__":
    principal()
