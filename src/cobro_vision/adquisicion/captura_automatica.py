"""Control temporal de la captura automatica."""

from time import monotonic

import cv2
import numpy as np


class CapturaAutomatica:

    def __init__(
        self,
        estabilidad_s=2.0,
        umbral_movimiento=0.02,
        umbral_presencia=0.08,
        retiro_s=1.0,
    ):
        self.estabilidad_s = estabilidad_s
        self.umbral_movimiento = umbral_movimiento
        self.umbral_presencia = umbral_presencia
        self.retiro_s = retiro_s

        self.estado = "calibrando"
        self.fondo = None
        self.anterior = None
        self.referencia_estable = None
        self.inicio_estabilidad = None
        self.inicio_retiro = None

    @staticmethod
    def reducir(frame):
        imagen = cv2.resize(frame, (160, 120))
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(imagen, (5, 5), 0)

    @staticmethod
    def diferencia(a, b):
        diferencia = cv2.absdiff(a, b)
        return float(np.mean(diferencia > 20))

    def calibrar(self, frame):
        self.fondo = self.reducir(frame)
        self.anterior = None
        self.referencia_estable = None
        self.inicio_estabilidad = None
        self.inicio_retiro = None
        self.estado = "esperando"

    def bloquear(self):
        self.estado = "bloqueado"

    def reiniciar(self):
        self.estado = "esperando"
        self.anterior = None
        self.referencia_estable = None
        self.inicio_estabilidad = None
        self.inicio_retiro = None

    def esperar_retiro(self):
        self.estado = "esperando_retiro"
        self.inicio_retiro = None

    def actualizar(self, frame, ahora=None):

        ahora = monotonic() if ahora is None else ahora

        if self.fondo is None:
            return False

        if self.estado == "bloqueado":
            return False

        actual = self.reducir(frame)

        presencia = (
            self.diferencia(actual, self.fondo)
            >= self.umbral_presencia
        )

        if self.estado == "esperando_retiro":

            if presencia:
                self.inicio_retiro = None

            elif self.inicio_retiro is None:
                self.inicio_retiro = ahora

            elif ahora - self.inicio_retiro >= self.retiro_s:
                self.estado = "esperando"
                self.anterior = None
                self.referencia_estable = None
                self.inicio_estabilidad = None

            return False

        if not presencia:
            self.estado = "esperando"
            self.anterior = None
            self.referencia_estable = None
            self.inicio_estabilidad = None
            return False

        if self.anterior is None:
            self.anterior = actual
            self.referencia_estable = actual.copy()
            self.inicio_estabilidad = ahora
            self.estado = "estabilizando"
            return False

        movimiento = self.diferencia(
            actual, self.anterior
        )

        desplazamiento = self.diferencia(
            actual, self.referencia_estable
        )

        self.anterior = actual

        if (
            movimiento > self.umbral_movimiento
            or desplazamiento > self.umbral_movimiento
        ):
            self.inicio_estabilidad = ahora
            self.referencia_estable = actual.copy()
            self.estado = "estabilizando"
            return False

        if ahora - self.inicio_estabilidad >= self.estabilidad_s:
            self.estado = "bloqueado"
            return True

        return False
