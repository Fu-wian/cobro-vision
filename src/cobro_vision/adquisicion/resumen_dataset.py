"""Cobertura por clase de las anotaciones registradas en el manifiesto."""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from cobro_vision.adquisicion.gestor_dataset import GestorDataset
from cobro_vision.configuracion import Configuracion
from cobro_vision.dominio.errores import EtiquetaInvalida
from cobro_vision.negocio.catalogo import CatalogoAlimentos


@dataclass
class _ConteoClase:
    platos: set[str] = field(default_factory=set)
    train: set[str] = field(default_factory=set)
    test: set[str] = field(default_factory=set)
    fotos: set[str] = field(default_factory=set)
    poligonos: int = 0


def resumir_dataset(config: Configuracion, catalogo: CatalogoAlimentos) -> pd.DataFrame:
    """Cuenta platos únicos, fotos y formas por clase sin escribir en el dataset."""
    ruta = Path(config.seccion("rutas")["manifiesto"])
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No existe el manifiesto: {ruta}. Ejecute construir-manifiesto."
        )
    conteos = {clase: _ConteoClase() for clase in catalogo.clases_validas()}
    muestras = GestorDataset(config, catalogo).cargar_muestras()
    # TODO(equipo): se resume cobertura visible de los JSON, sin añadir overrides;
    # cada rectángulo admitido por LabelMe cuenta como un polígono de cuatro puntos.
    for muestra in muestras:
        for forma in muestra.etiquetado.formas:
            if forma.etiqueta not in conteos:
                raise EtiquetaInvalida(
                    f"{muestra.id_foto}: etiqueta fuera del catálogo: {forma.etiqueta!r}."
                )
            conteo = conteos[forma.etiqueta]
            conteo.platos.add(muestra.plato_id)
            if muestra.split == "train":
                conteo.train.add(muestra.plato_id)
            elif muestra.split == "test":
                conteo.test.add(muestra.plato_id)
            conteo.fotos.add(muestra.id_foto)
            conteo.poligonos += 1
    filas = []
    for clase, conteo in conteos.items():
        marcas = []
        if len(conteo.train) < 3:
            marcas.append("FALTA")
        if len(conteo.platos) < 25:
            marcas.append("POCO")
        filas.append(
            {
                "id_clase": clase,
                "platos_total": len(conteo.platos),
                "platos_train": len(conteo.train),
                "platos_test": len(conteo.test),
                "fotos": len(conteo.fotos),
                "poligonos": conteo.poligonos,
                "estado": " ".join(marcas),
            }
        )
    return pd.DataFrame(
        filas,
        columns=[
            "id_clase",
            "platos_total",
            "platos_train",
            "platos_test",
            "fotos",
            "poligonos",
            "estado",
        ],
    )
