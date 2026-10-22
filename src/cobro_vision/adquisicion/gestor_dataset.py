"""Manifiesto y división reproducible por plato físico, nunca por fotografía."""

import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cobro_vision.adquisicion.lector_labelme import LectorLabelMe
from cobro_vision.configuracion import Configuracion
from cobro_vision.dominio.errores import EtiquetaInvalida
from cobro_vision.dominio.modelos import ID_FONDO, Muestra
from cobro_vision.negocio.catalogo import CatalogoAlimentos

_REGISTRO = logging.getLogger(__name__)
_COLUMNAS = [
    "id_foto",
    "ruta_imagen",
    "plato_id",
    "sesion",
    "fecha",
    "split",
    "contenido_esperado_override",
]
_EXTENSIONES = {".jpg", ".jpeg", ".png"}


def _escanear_archivos(carpeta: Path) -> tuple[list[Path], list[Path]]:
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de imágenes: {carpeta}")
    archivos = sorted(
        (ruta.resolve() for ruta in carpeta.rglob("*") if ruta.is_file()),
        key=lambda ruta: ruta.as_posix(),
    )
    return (
        [ruta for ruta in archivos if ruta.suffix.lower() in _EXTENSIONES],
        [ruta for ruta in archivos if ruta.suffix.lower() == ".json"],
    )


def _identidad(ruta_relativa: Path) -> tuple[str, str, str]:
    if (
        ruta_relativa.is_absolute()
        or len(ruta_relativa.parts) != 2
        or any(parte in (".", "..") for parte in ruta_relativa.parts)
        or ruta_relativa.suffix.lower() not in _EXTENSIONES
    ):
        raise EtiquetaInvalida(
            f"Ruta fuera del formato <SESION>/<PLATO>_<n>: {ruta_relativa}"
        )
    plato, separador, toma = ruta_relativa.stem.partition("_")
    if not plato or not separador or not toma:
        raise EtiquetaInvalida(f"Nombre fuera del formato <PLATO>_<n>: {ruta_relativa}")
    return ruta_relativa.with_suffix("").as_posix(), plato, ruta_relativa.parts[0]


def _splits_por_plato(manifiesto: pd.DataFrame) -> dict[str, str]:
    asignados: dict[str, str] = {}
    for plato, grupo in manifiesto.groupby("plato_id", sort=True):
        conjuntos = set(grupo["split"]) - {""}
        if conjuntos - {"train", "test"}:
            raise EtiquetaInvalida(
                f"{plato}: split inválido; se admite train, test o vacío."
            )
        if len(conjuntos) > 1:
            raise EtiquetaInvalida(
                f"Fuga de datos: plato_id {plato} aparece en train y test."
            )
        asignados[plato] = next(iter(conjuntos), "")
    return asignados


class GestorDataset:
    """Conserva los splits y el contenido esperado registrados en el CSV."""

    def __init__(self, config: Configuracion, catalogo: CatalogoAlimentos) -> None:
        rutas = config.seccion("rutas")
        self._carpeta = Path(rutas["imagenes"]).resolve()
        self._ruta_manifiesto = Path(rutas["manifiesto"])
        self._catalogo = catalogo
        self._lector = LectorLabelMe()

    def construir_manifiesto(self) -> pd.DataFrame:
        """Escanea las fotos y agrega registros sin cambiar splits ya asignados."""
        anterior = self._leer_manifiesto()
        asignados = _splits_por_plato(anterior)
        registros = {fila["id_foto"]: fila for fila in anterior.to_dict("records")}
        imagenes, archivos_json = _escanear_archivos(self._carpeta)
        por_nombre = {ruta.with_suffix(""): ruta for ruta in archivos_json}
        vistos: set[str] = set()
        for ruta in imagenes:
            relativa = ruta.relative_to(self._carpeta)
            id_foto, plato, sesion = _identidad(relativa)
            if id_foto in vistos:
                raise EtiquetaInvalida(
                    f"id_foto duplicado (varias extensiones): {id_foto}"
                )
            vistos.add(id_foto)
            existente = registros.get(id_foto, {})
            registros[id_foto] = {
                "id_foto": id_foto,
                "ruta_imagen": relativa.as_posix(),
                "plato_id": plato,
                "sesion": sesion,
                "fecha": datetime.fromtimestamp(ruta.stat().st_mtime)
                .date()
                .isoformat(),
                "split": existente.get("split") or asignados.get(plato, ""),
                "contenido_esperado_override": existente.get(
                    "contenido_esperado_override", ""
                ),
            }
            if ruta.with_suffix("") not in por_nombre:
                _REGISTRO.warning(
                    "Imagen sin JSON (se omite al cargar muestras): %s", ruta
                )
        # TODO(equipo): se conservan también filas de fotos retiradas para no perder
        # splits ni overrides; cargar_muestras informa si falta una foto etiquetada.
        manifiesto = pd.DataFrame(registros.values(), columns=_COLUMNAS)
        asignados = _splits_por_plato(manifiesto)
        manifiesto["split"] = manifiesto["plato_id"].map(asignados)
        self._guardar(manifiesto)
        return manifiesto.sort_values("id_foto").reset_index(drop=True)

    def dividir(self, fraccion_prueba: float, semilla: int) -> None:
        """Asigna todas las fotos de cada plato al mismo conjunto con la semilla dada."""
        if not math.isfinite(fraccion_prueba) or not 0 < fraccion_prueba < 1:
            raise ValueError(
                "La fracción de prueba debe estar entre 0 y 1, sin extremos."
            )
        if type(semilla) is not int or not 0 <= semilla < 2**32:
            raise ValueError("semilla debe ser un entero entre 0 y 2**32 - 1.")
        manifiesto = self._obtener_manifiesto()
        platos = sorted(set(manifiesto["plato_id"]))
        if len(platos) < 2:
            raise EtiquetaInvalida(
                "Se necesitan al menos 2 plato_id para dividir train/test."
            )
        muestras = self._muestras_desde(manifiesto)
        # TODO(equipo): se redondea la cantidad de platos de test hacia arriba,
        # reservando al menos uno para train cuando el dataset es pequeño.
        cantidad = min(len(platos) - 1, math.ceil(len(platos) * fraccion_prueba))
        prueba = set(np.random.default_rng(semilla).permutation(platos)[:cantidad])
        manifiesto["split"] = manifiesto["plato_id"].map(
            lambda plato: "test" if plato in prueba else "train"
        )
        self._guardar(manifiesto)
        for conjunto in ("train", "test"):
            presentes = {
                forma.etiqueta
                for muestra in muestras
                if ("test" if muestra.plato_id in prueba else "train") == conjunto
                for forma in muestra.etiquetado.formas
            }
            for clase in sorted(set(self._catalogo.clases_validas()) - presentes):
                _REGISTRO.warning("Clase %s sin ejemplos en %s.", clase, conjunto)

    def cargar_muestras(self, split: str | None = None) -> list[Muestra]:
        """Lee fotos etiquetadas y comprueba todos los grupos antes de filtrar el split."""
        if split not in (None, "train", "test"):
            raise ValueError("split debe ser train, test o None.")
        manifiesto = self._obtener_manifiesto()
        if split is not None:
            manifiesto = manifiesto[manifiesto["split"] == split]
        return self._muestras_desde(manifiesto)

    def contenido_esperado(self, muestra: Muestra) -> set[str]:
        """Usa el override completo si existe; de otro modo, las etiquetas visibles."""
        manifiesto = self._obtener_manifiesto()
        filas = manifiesto[manifiesto["id_foto"] == muestra.id_foto]
        override = "" if filas.empty else filas.iloc[0]["contenido_esperado_override"]
        clases = (
            {valor.strip() for valor in override.split(";") if valor.strip()}
            if override.strip()
            else {forma.etiqueta for forma in muestra.etiquetado.formas}
        )
        desconocidas = clases - set(self._catalogo.clases_validas())
        if desconocidas:
            raise EtiquetaInvalida(
                f"{muestra.id_foto}: clases fuera del catálogo: {sorted(desconocidas)}"
            )
        return clases - {ID_FONDO}

    def _obtener_manifiesto(self) -> pd.DataFrame:
        return (
            self._leer_manifiesto()
            if self._ruta_manifiesto.exists()
            else self.construir_manifiesto()
        )

    def _leer_manifiesto(self) -> pd.DataFrame:
        if not self._ruta_manifiesto.exists():
            return pd.DataFrame(columns=_COLUMNAS)
        manifiesto = pd.read_csv(
            self._ruta_manifiesto,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        faltantes = set(_COLUMNAS[:-1]) - set(manifiesto.columns)
        if faltantes:
            raise EtiquetaInvalida(
                f"Manifiesto sin columnas obligatorias: {sorted(faltantes)}"
            )
        if "contenido_esperado_override" not in manifiesto:
            manifiesto["contenido_esperado_override"] = ""
        if manifiesto["id_foto"].duplicated().any():
            raise EtiquetaInvalida("El manifiesto contiene id_foto duplicados.")
        for fila in manifiesto.to_dict("records"):
            identidad = _identidad(Path(fila["ruta_imagen"].replace("\\", "/")))
            if identidad != (fila["id_foto"], fila["plato_id"], fila["sesion"]):
                raise EtiquetaInvalida(
                    f"{fila['id_foto']}: identidad inconsistente en el manifiesto."
                )
        _splits_por_plato(manifiesto)
        return manifiesto[_COLUMNAS]

    def _guardar(self, manifiesto: pd.DataFrame) -> None:
        _splits_por_plato(manifiesto)
        self._ruta_manifiesto.parent.mkdir(parents=True, exist_ok=True)
        temporal = self._ruta_manifiesto.with_suffix(".csv.tmp")
        manifiesto.sort_values("id_foto")[_COLUMNAS].to_csv(
            temporal, index=False, encoding="utf-8", lineterminator="\n"
        )
        temporal.replace(self._ruta_manifiesto)

    def _muestras_desde(self, manifiesto: pd.DataFrame) -> list[Muestra]:
        _, archivos_json = _escanear_archivos(self._carpeta)
        por_nombre = {ruta.with_suffix(""): ruta for ruta in archivos_json}
        muestras = []
        for fila in manifiesto.sort_values("id_foto").to_dict("records"):
            ruta = self._carpeta / Path(fila["ruta_imagen"].replace("\\", "/"))
            ruta_json = por_nombre.get(ruta.with_suffix(""))
            if ruta_json is None:
                _REGISTRO.warning(
                    "Imagen sin JSON (se omite al cargar muestras): %s", ruta
                )
                continue
            etiquetado = self._lector.leer(ruta_json)
            if etiquetado.ruta_imagen != ruta.resolve():
                raise EtiquetaInvalida(
                    f"{ruta_json}: imagePath no corresponde a {ruta.name}."
                )
            muestras.append(
                Muestra(
                    fila["id_foto"],
                    fila["plato_id"],
                    fila["sesion"],
                    fila["split"],
                    ruta,
                    etiquetado,
                )
            )
        return muestras
