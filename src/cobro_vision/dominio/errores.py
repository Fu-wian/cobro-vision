"""Errores reconocibles por la CLI y los futuros controladores."""


class CobroVisionError(Exception):
    """Error propio del prototipo."""


class ConfiguracionInvalida(CobroVisionError):
    """La configuración contiene un campo inválido."""


class EtiquetaInvalida(CobroVisionError):
    """Una etiqueta no cumple el formato esperado."""


class PlatoNoDetectado(CobroVisionError):
    """No se pudo localizar un plato válido."""


class ClaseDesconocida(CobroVisionError):
    """La clase no existe en el catálogo activo."""


class ModeloIncompatible(CobroVisionError):
    """El modelo no corresponde a las características configuradas."""
