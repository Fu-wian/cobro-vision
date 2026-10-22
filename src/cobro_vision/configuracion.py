"""Carga, validación y huella reproducible de la configuración YAML."""

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import yaml

from cobro_vision.dominio.errores import ConfiguracionInvalida


def _campo(datos: dict, ruta: str) -> object:
    valor = datos
    for parte in ruta.split("."):
        if not isinstance(valor, dict) or parte not in valor:
            raise ConfiguracionInvalida(f"{ruta}: falta el campo obligatorio.")
        valor = valor[parte]
    return valor


def _exigir(condicion: bool, campo: str, mensaje: str) -> None:
    if not condicion:
        raise ConfiguracionInvalida(f"{campo}: {mensaje}")


def _numero(
    valor: object,
    campo: str,
    minimo: float = 0,
    maximo: float = math.inf,
    entero: bool = False,
    positivo: bool = False,
) -> None:
    _exigir(
        type(valor) is int if entero else type(valor) in (int, float),
        campo,
        "se esperaba un entero." if entero else "se esperaba un número.",
    )
    _exigir(
        (type(valor) is int or math.isfinite(valor))
        and minimo <= valor <= maximo
        and (not positivo or valor > 0),
        campo,
        "valor fuera del rango permitido.",
    )


def _lista(valor: object, campo: str, longitud: int | None = None) -> None:
    _exigir(
        isinstance(valor, list) and bool(valor),
        campo,
        "se esperaba una lista no vacía.",
    )
    if longitud is not None:
        _exigir(len(valor) == longitud, campo, f"se esperaban {longitud} elementos.")


def _validar(datos: dict) -> None:
    secciones = (
        "rutas",
        "imagen",
        "plato",
        "segmentacion",
        "clasificador",
        "entrenamiento",
        "posproceso",
        "cobro",
        "camara",
        "ui",
        "imagen.filtro",
        "imagen.correccion_color",
        "plato.umbral",
        "segmentacion.ventanas",
        "clasificador.busqueda",
        "cobro.por_plato",
    )
    for campo in secciones:
        _exigir(
            isinstance(_campo(datos, campo), dict), campo, "se esperaba un diccionario."
        )
    for nombre in (
        "imagenes",
        "manifiesto",
        "catalogo",
        "semilla_catalogo",
        "modelos",
        "modelo_actual",
        "salidas",
    ):
        valor = _campo(datos, f"rutas.{nombre}")
        _exigir(
            isinstance(valor, str) and bool(valor.strip()),
            f"rutas.{nombre}",
            "ruta vacía o inválida.",
        )
    for campo in ("camara.carpeta", "ui.moneda"):
        valor = _campo(datos, campo)
        _exigir(
            isinstance(valor, str) and bool(valor.strip()),
            campo,
            "se esperaba texto no vacío.",
        )

    opciones = {
        "imagen.filtro.tipo": ("mediana", "gaussiano", "ninguno"),
        "imagen.correccion_color.metodo": ("gray_world", "tarjeta_gris", "ninguna"),
        "plato.metodo": ("hough", "umbral", "imagen_completa"),
        "segmentacion.clase": ("SegmentadorSuperpixeles", "SegmentadorVentanas"),
        "clasificador.tipo": ("falso", "svm"),
        "clasificador.kernel": ("linear", "poly", "rbf", "sigmoid"),
        "entrenamiento.modo": ("poligonos", "segmentacion", "ambos"),
        "cobro.politica": ("CobroPorComponente", "CobroPorPlato"),
        "camara.clase": ("CamaraArchivo", "CamaraUSB"),
        "ui.tema": ("system", "light", "dark"),
    }
    for campo, permitidos in opciones.items():
        _exigir(
            _campo(datos, campo) in permitidos,
            campo,
            f"valores permitidos: {permitidos}.",
        )

    enteros = {
        "semilla": (0, 2**32 - 1),
        "imagen.lado_mayor": (1, math.inf),
        "imagen.filtro.tamano": (1, math.inf),
        "segmentacion.n_segmentos": (1, math.inf),
        "segmentacion.area_minima_px": (1, math.inf),
        "segmentacion.ventanas.tamano": (1, math.inf),
        "segmentacion.ventanas.paso": (1, math.inf),
        "plato.umbral.s_max": (0, 255),
        "plato.umbral.v_min": (0, 255),
        "clasificador.busqueda.folds": (2, math.inf),
        "entrenamiento.parches_por_poligono": (0, math.inf),
        "entrenamiento.radio_parche": (1, math.inf),
        "entrenamiento.min_instancias_por_clase": (1, math.inf),
        "camara.indice": (0, math.inf),
        "ui.vista_previa_ms": (1, math.inf),
    }
    for campo, (minimo, maximo) in enteros.items():
        _numero(_campo(datos, campo), campo, minimo, maximo, entero=True)
    _exigir(
        _campo(datos, "imagen.filtro.tamano") % 2 == 1,
        "imagen.filtro.tamano",
        "el tamaño del filtro debe ser impar.",
    )
    for campo in ("segmentacion.compacidad", "clasificador.C"):
        _numero(_campo(datos, campo), campo, positivo=True)
    for campo in ("segmentacion.sigma", "clasificador.umbral_confianza"):
        _numero(_campo(datos, campo), campo)
    for campo in (
        "plato.radio_min_frac",
        "plato.radio_max_frac",
        "plato.reducir_radio",
        "plato.area_minima_frac",
        "entrenamiento.pureza_minima",
    ):
        _numero(_campo(datos, campo), campo, maximo=1, positivo=True)
    for campo in ("posproceso.componente_minimo_frac", "posproceso.clase_minima_frac"):
        _numero(_campo(datos, campo), campo, maximo=1)
    _exigir(
        _campo(datos, "plato.radio_min_frac") < _campo(datos, "plato.radio_max_frac"),
        "plato.radio_max_frac",
        "debe superar radio_min_frac.",
    )
    limite = _campo(datos, "posproceso.max_proteinas")
    if limite is not None:
        _numero(limite, "posproceso.max_proteinas", entero=True)
    _exigir(
        type(_campo(datos, "ui.exigir_revision")) is bool,
        "ui.exigir_revision",
        "se esperaba bool.",
    )
    dimensiones = _campo(datos, "ui.tamano_ventana")
    _lista(dimensiones, "ui.tamano_ventana", 2)
    for valor in dimensiones:
        _numero(valor, "ui.tamano_ventana", entero=True, positivo=True)

    def validar_gamma(valor: object, campo: str) -> None:
        """Admite gamma automático o un número positivo."""
        if isinstance(valor, str):
            _exigir(
                valor in ("scale", "auto"),
                campo,
                "se esperaba scale, auto o un número positivo.",
            )
        else:
            _numero(valor, campo, positivo=True)

    validar_gamma(_campo(datos, "clasificador.gamma"), "clasificador.gamma")
    for nombre in ("C", "gamma"):
        campo = f"clasificador.busqueda.{nombre}"
        valores = _campo(datos, campo)
        _lista(valores, campo)
        for valor in valores:
            if nombre == "gamma":
                validar_gamma(valor, campo)
            else:
                _numero(valor, campo, positivo=True)
    pesos = _campo(datos, "clasificador.class_weight")
    if isinstance(pesos, dict):
        for nombre, valor in pesos.items():
            _exigir(
                isinstance(nombre, str),
                "clasificador.class_weight",
                "las clases deben ser texto.",
            )
            _numero(valor, f"clasificador.class_weight.{nombre}", positivo=True)
    else:
        _exigir(
            pesos in (None, "balanced"),
            "clasificador.class_weight",
            "se esperaba balanced, null o un diccionario.",
        )

    precios = _campo(datos, "cobro.por_plato.precios_por_proteina_centavos")
    campo = "cobro.por_plato.precios_por_proteina_centavos"
    _exigir(isinstance(precios, dict), campo, "se esperaba un diccionario de centavos.")
    for nombre, precio in precios.items():
        _exigir(isinstance(nombre, str) and bool(nombre), campo, "id_clase inválido.")
        _numero(precio, f"{campo}.{nombre}", entero=True)

    extractores = _campo(datos, "caracteristicas")
    _lista(extractores, "caracteristicas")
    for indice, extractor in enumerate(extractores):
        prefijo = f"caracteristicas[{indice}]"
        _exigir(isinstance(extractor, dict), prefijo, "se esperaba un diccionario.")
        clase = _campo(extractor, "clase")
        _exigir(
            clase in ("ExtractorHistogramaColor", "ExtractorTextura"),
            f"{prefijo}.clase",
            "extractor desconocido.",
        )
        if clase == "ExtractorHistogramaColor":
            _exigir(
                _campo(extractor, "espacio") in ("hsv", "lab"),
                f"{prefijo}.espacio",
                "se esperaba hsv o lab.",
            )
            bins = _campo(extractor, "bins")
            _lista(bins, f"{prefijo}.bins", 3)
            for valor in bins:
                _numero(valor, f"{prefijo}.bins", entero=True, positivo=True)
            booleanos = ("momentos",)
        else:
            for nombre in ("radio", "puntos"):
                _numero(
                    _campo(extractor, f"lbp.{nombre}"),
                    f"{prefijo}.lbp.{nombre}",
                    entero=True,
                    positivo=True,
                )
            booleanos = ("gradiente", "glcm")
        for nombre in booleanos:
            _exigir(
                type(_campo(extractor, nombre)) is bool,
                f"{prefijo}.{nombre}",
                "se esperaba bool.",
            )
    if _campo(datos, "imagen.correccion_color.metodo") == "tarjeta_gris":
        # TODO(equipo): El esquema no nombra el ROI; se usa correccion_color.roi [x,y,w,h].
        roi = _campo(datos, "imagen.correccion_color.roi")
        _lista(roi, "imagen.correccion_color.roi", 4)
        for indice, valor in enumerate(roi):
            _numero(
                valor,
                "imagen.correccion_color.roi",
                entero=True,
                minimo=0 if indice < 2 else 1,
            )


@dataclass(frozen=True)
class Configuracion:
    """Configuración validada sin cambios accidentales a través de seccion."""

    _datos: dict

    @classmethod
    def cargar(cls, ruta: Path = Path("config.yaml")) -> "Configuracion":
        """Lee el esquema completo y señala el campo de cualquier error."""
        try:
            with ruta.open(encoding="utf-8-sig") as archivo:
                datos = yaml.safe_load(archivo)
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise ConfiguracionInvalida(
                f"{ruta}: no se pudo leer la configuración: {error}"
            ) from error
        _exigir(isinstance(datos, dict), "config", "se esperaba un diccionario YAML.")
        _validar(datos)
        try:
            json.dumps(datos, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ConfiguracionInvalida(
                f"config: contiene valores no serializables: {error}"
            ) from error
        return cls(deepcopy(datos))

    @property
    def semilla(self) -> int:
        """Devuelve la semilla compartida por el proyecto."""
        return self._datos["semilla"]

    def seccion(self, nombre: str) -> dict:
        """Devuelve una copia de una sección de configuración."""
        valor = _campo(self._datos, nombre)
        if nombre == "caracteristicas":
            # TODO(equipo): seccion promete dict, pero caracteristicas es una lista;
            # se expone como {"extractores": [...]} sin cambiar la firma acordada.
            return {"extractores": deepcopy(valor)}
        _exigir(
            isinstance(valor, dict), nombre, "no es una sección de tipo diccionario."
        )
        return deepcopy(valor)

    def hash(self, secciones: list[str] | None = None) -> str:
        """Obtiene 12 hexadecimales SHA256 independientes del orden de las claves."""
        datos = (
            self._datos
            if secciones is None
            else {
                nombre: _campo(self._datos, nombre) for nombre in sorted(set(secciones))
            }
        )
        contenido = json.dumps(
            datos,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(contenido.encode("utf-8")).hexdigest()[:12]
