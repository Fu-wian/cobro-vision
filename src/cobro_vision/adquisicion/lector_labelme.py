"""Lectura de LabelMe en las coordenadas de la imagen original."""

import json
import math
from pathlib import Path

import cv2

from cobro_vision.configuracion import Configuracion
from cobro_vision.dominio.errores import ConfiguracionInvalida, EtiquetaInvalida
from cobro_vision.dominio.modelos import Etiquetado, Forma
from cobro_vision.negocio.catalogo import CatalogoAlimentos


def _cargar_alias(
    config: Configuracion, catalogo: CatalogoAlimentos | None = None
) -> dict[str, str]:
    alias = config.seccion("etiquetas")["alias"]
    if alias:
        if catalogo is None:
            catalogo = CatalogoAlimentos(Path(config.seccion("rutas")["catalogo"]))
        clases = set(catalogo.clases_validas())
        for origen, destino in alias.items():
            if destino not in clases:
                raise ConfiguracionInvalida(
                    f"etiquetas.alias.{origen}: destino fuera del catálogo activo: {destino!r}."
                )
    return alias


def _leer_etiquetado(
    ruta_json: Path, alias: dict[str, str] | None = None
) -> tuple[Etiquetado | None, list[str]]:
    """Acumula problemas independientes para que el validador los informe juntos."""
    errores: list[str] = []

    def registrar(mensaje: str) -> None:
        errores.append(f"{ruta_json}: {mensaje}")

    try:
        datos = json.loads(ruta_json.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as error:
        registrar(f"no se pudo leer el JSON: {error}")
        return None, errores
    if not isinstance(datos, dict):
        registrar("se esperaba un objeto JSON.")
        return None, errores
    nombre = datos.get("imagePath")
    if not isinstance(nombre, str) or not nombre.strip():
        registrar("imagePath debe indicar la imagen original.")
        return None, errores
    ruta_imagen = (ruta_json.parent / nombre.replace("\\", "/")).resolve()
    ancho, alto = datos.get("imageWidth"), datos.get("imageHeight")
    dimensiones_validas = all(
        type(valor) is int and valor > 0 for valor in (ancho, alto)
    )
    if not dimensiones_validas:
        registrar("imageWidth e imageHeight deben ser enteros positivos.")
    if not ruta_imagen.is_file():
        registrar(f"JSON sin imagen: {ruta_imagen}")
    else:
        # IMREAD_COLOR aplica EXIF, igual que la lectura de fotos del pipeline.
        imagen = cv2.imread(str(ruta_imagen), cv2.IMREAD_COLOR)
        if imagen is None:
            registrar(f"no se pudo decodificar la imagen: {ruta_imagen}")
        elif dimensiones_validas and imagen.shape[:2] != (alto, ancho):
            registrar(
                f"dimensiones distintas: imageWidth/imageHeight={ancho}x{alto}; "
                f"cv2.imread={imagen.shape[1]}x{imagen.shape[0]}. "
                "Revise la orientación EXIF o si la foto fue redimensionada."
            )
    formas: list[Forma] = []
    originales = datos.get("shapes")
    if not isinstance(originales, list):
        registrar("shapes debe ser una lista.")
        return None, errores
    for numero, original in enumerate(originales, start=1):
        prefijo = f"forma {numero}"
        if not isinstance(original, dict):
            registrar(f"{prefijo}: se esperaba un objeto.")
            continue
        etiqueta, tipo = original.get("label"), original.get("shape_type")
        if not isinstance(etiqueta, str) or not etiqueta.strip():
            registrar(f"{prefijo}: label debe ser texto no vacío.")
            continue
        if tipo not in ("polygon", "rectangle"):
            registrar(f"{prefijo}: tipo de forma no soportado: {tipo!r}.")
        coordenadas = original.get("points")
        if not isinstance(coordenadas, list):
            registrar(f"{prefijo}: points debe ser una lista de pares numéricos.")
            continue
        puntos: list[tuple[float, float]] = []
        for indice, punto in enumerate(coordenadas, start=1):
            if (
                not isinstance(punto, list)
                or len(punto) != 2
                or any(
                    type(valor) not in (int, float) or not math.isfinite(valor)
                    for valor in punto
                )
            ):
                registrar(f"{prefijo}, punto {indice}: coordenadas no válidas.")
                continue
            x, y = map(float, punto)
            puntos.append((x, y))
            if dimensiones_validas and not (0 <= x < ancho and 0 <= y < alto):
                registrar(f"{prefijo}, punto {indice}: fuera de la imagen ({x}, {y}).")
        if tipo == "polygon" and len(puntos) < 3:
            registrar(f"{prefijo}: polígono con menos de 3 puntos.")
        elif tipo == "rectangle":
            if len(puntos) != 2:
                registrar(f"{prefijo}: rectangle requiere 2 puntos.")
            else:
                (x0, y0), (x1, y1) = puntos
                puntos = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        formas.append(Forma((alias or {}).get(etiqueta, etiqueta), tipo, puntos))
    if not dimensiones_validas:
        return None, errores
    return Etiquetado(ruta_imagen, ruta_json, ancho, alto, formas), errores


class LectorLabelMe:
    """Lee polígonos y rectángulos, sin usar imageData ni alterar la escala."""

    # TODO(equipo): el contrato no fija constructor; la configuración opcional
    # permite inyectar alias y conserva el uso existente de LectorLabelMe().
    def __init__(
        self,
        config: Configuracion | None = None,
        catalogo: CatalogoAlimentos | None = None,
    ) -> None:
        self._alias = _cargar_alias(config, catalogo) if config is not None else {}

    def leer(self, ruta_json: Path) -> Etiquetado:
        """Valida estructura, puntos y dimensiones de la foto orientada por EXIF."""
        etiquetado, errores = _leer_etiquetado(ruta_json, self._alias)
        if errores:
            raise EtiquetaInvalida("\n".join(errores))
        assert etiquetado is not None
        return etiquetado
