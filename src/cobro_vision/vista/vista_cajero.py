"""Ventana CustomTkinter: widgets en Tk y trabajo del controlador en un hilo."""

import logging
from time import monotonic

from cobro_vision.adquisicion.captura_automatica import CapturaAutomatica
from cobro_vision.adquisicion.camara import CamaraUSB
from collections.abc import Callable
from functools import partial
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from tkinter import filedialog

import customtkinter as ctk
import numpy as np
from PIL import Image

from cobro_vision.configuracion import Configuracion
from cobro_vision.controlador.controlador import ControladorTransaccion
from cobro_vision.dominio.modelos import Transaccion
from cobro_vision.vista.presentador import (
    clases_visibles,
    etiquetas_alimentos,
    filas_recibo,
    mensajes_alerta,
    opciones_correccion,
    puede_confirmar,
    puede_editar,
    texto_explicacion,
    texto_tiempos,
    texto_total,
)
from cobro_vision.vista.render import ajustar_a_caja, dibujar_regiones, paleta_clases


def _ejecutar_trabajo(
    tarea: Callable[[], Transaccion], cola: Queue[Transaccion | Exception]
) -> None:
    # Este hilo solo recibe la operación y la cola: no accede a la ventana ni a Tk.
    try:
        cola.put(tarea())
    except Exception as error:
        logging.getLogger(__name__).exception("Falló la operación del cajero")
        cola.put(error)


class VistaCajero(ctk.CTk):
    """Muestra una transacción y delega su procesamiento y ajustes al controlador."""

    def __init__(
        self, controlador: ControladorTransaccion, config: Configuracion
    ) -> None:
        self._ui = config.seccion("ui")
        self._automatico = (
            isinstance(controlador.camara, CamaraUSB)
            and self._ui.get("captura_automatica", True)
        )
        self._pausada = False
        self._consulta_camara = None
        self._inicio_camara = monotonic()
        self._captura_auto = CapturaAutomatica(
            estabilidad_s=0.8,
            umbral_movimiento=0.05,
            umbral_presencia=0.08,
            retiro_s=1.0,
        )
        self._ultimo_frame = None
        self._origen_recibo = None
        ctk.set_appearance_mode(self._ui["tema"])
        super().__init__()
        self.controlador = controlador
        self.transaccion: Transaccion | None = None
        self._ocupada = False
        self._cerrando = False
        self._destruida = False
        self._cola: Queue[Transaccion | Exception] = Queue()
        self._hilo: Thread | None = None
        self._consulta: str | None = None
        self._con_marcas = True
        self._imagen_bgr: np.ndarray | None = None
        transparente = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        self._imagen_vacia = ctk.CTkImage(
            light_image=transparente, dark_image=transparente, size=(1, 1)
        )
        self._imagen_tk: ctk.CTkImage = self._imagen_vacia
        self._controles_filas: list[ctk.CTkBaseClass] = []
        self._etiquetas = etiquetas_alimentos(controlador.catalogo)
        self._ids_opciones = {texto: clase for clase, texto in self._etiquetas.items()}
        self._paleta = paleta_clases(controlador.catalogo.clases_validas())
        self.title("Cobro Vision · Cajero")
        ancho, alto = self._ui["tamano_ventana"]
        self.geometry(f"{ancho}x{alto}")
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        izquierda = ctk.CTkFrame(self)
        izquierda.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")
        izquierda.grid_columnconfigure(0, weight=1)
        izquierda.grid_rowconfigure(1, weight=1)
        self._desde_archivo = True

        barra = ctk.CTkFrame(izquierda)
        barra.grid(row=0, column=0, pady=10)

        self._boton_escanear = ctk.CTkButton(
            barra,
            text="Abrir foto...",
            command=self.al_escanear
        )
        self._boton_escanear.pack(side="left", padx=4)

        if self._automatico:
            self._boton_pausa = ctk.CTkButton(
                barra,
                text="Pausar camara",
                command=self._alternar_camara
            )
            self._boton_pausa.pack(side="left", padx=4)

            self._boton_calibrar = ctk.CTkButton(
                barra,
                text="Calibrar fondo",
                command=self._calibrar_fondo
            )
            self._boton_calibrar.pack(side="left", padx=4)

        self._panel_imagen = ctk.CTkFrame(izquierda)
        self._panel_imagen.grid(row=1, column=0, padx=10, sticky="nsew")
        self._panel_imagen.grid_columnconfigure(0, weight=1)
        self._panel_imagen.grid_rowconfigure(0, weight=1)
        self._foto = ctk.CTkLabel(
            self._panel_imagen, image=self._imagen_tk, text="Abra una foto del plato"
        )
        self._foto.grid(row=0, column=0)
        self._panel_imagen.bind("<Configure>", lambda evento: self._redibujar_foto())
        self._leyenda = ctk.CTkFrame(izquierda)
        self._leyenda.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        self._boton_marcas = ctk.CTkButton(
            izquierda, text="Mostrar / ocultar regiones", command=self._alternar_marcas
        )
        self._boton_marcas.grid(row=3, column=0, pady=5)
        self._tiempos = ctk.CTkLabel(izquierda, text="", justify="left", anchor="w")
        self._tiempos.grid(row=4, column=0, padx=10, pady=5, sticky="ew")

        derecha = ctk.CTkFrame(self)
        derecha.grid(row=0, column=1, padx=(0, 12), pady=12, sticky="nsew")
        derecha.grid_columnconfigure(0, weight=1)
        derecha.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(derecha, text="Recibo", font=("Arial", 22)).grid(row=0, column=0)
        self._filas = ctk.CTkScrollableFrame(derecha)
        self._filas.grid(row=1, column=0, padx=10, pady=8, sticky="nsew")
        self._filas.grid_columnconfigure(0, weight=1)
        self._opcion_agregar = ctk.CTkOptionMenu(
            derecha, values=list(self._ids_opciones) or ["Sin alimentos"]
        )
        self._opcion_agregar.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        self._boton_agregar = ctk.CTkButton(
            derecha, text="Agregar ítem", command=self._agregar_seleccionado
        )
        self._boton_agregar.grid(row=3, column=0, pady=5)
        self._total = ctk.CTkLabel(
            derecha, text=texto_total(None, self._ui["moneda"]), font=("Arial", 30)
        )
        self._total.grid(row=4, column=0, pady=10)
        self._alerta = ctk.CTkLabel(
            derecha, text="", text_color=("#8a4500", "#ffbb66"), wraplength=350
        )
        self._alerta.grid(row=5, column=0, padx=10, pady=5, sticky="ew")
        self._boton_confirmar = ctk.CTkButton(
            derecha, text="Confirmar", command=self.al_confirmar
        )
        self._boton_confirmar.grid(row=6, column=0, pady=5)
        self._boton_nueva = ctk.CTkButton(
            derecha, text="Nueva transacción", command=self._nueva_transaccion
        )
        self._boton_nueva.grid(row=7, column=0, pady=5)
        self._boton_explicar = ctk.CTkButton(
            derecha, text="¿Por qué?", command=self._mostrar_explicacion
        )
        self._boton_explicar.grid(row=8, column=0, pady=10)
        self._actualizar_controles()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self._consulta = self.after(50, self._revisar_cola)

        if self._automatico:
            self.mostrar_alerta(
                "Mantenga vacia la superficie durante la calibracion."
            )
            self._consulta_camara = self.after(
                100, self._actualizar_camara
            )

    def mostrar_previsualizacion(self, imagen_bgr: np.ndarray) -> None:
        """Convierte BGR a RGB y conserva la referencia CTkImage en el hilo de Tk."""
        self._imagen_bgr = imagen_bgr
        self._redibujar_foto()

    def mostrar_regiones(self, t: Transaccion) -> None:
        """Dibuja la imagen procesada y la leyenda de las regiones visibles."""
        for widget in self._leyenda.winfo_children():
            widget.destroy()
        if t.imagen_procesada is None:
            return
        imagen = t.imagen_procesada
        if self._con_marcas:
            imagen = dibujar_regiones(imagen, t.regiones, self._paleta)
        self.mostrar_previsualizacion(imagen)
        for indice, clase in enumerate(clases_visibles(t)):
            azul, verde, rojo = self._paleta[clase]
            ctk.CTkLabel(
                self._leyenda,
                text=self._etiquetas.get(clase, clase),
                text_color=f"#{rojo:02x}{verde:02x}{azul:02x}",
            ).grid(row=indice // 2, column=indice % 2, padx=8, sticky="w")

    def mostrar_recibo(self, t: Transaccion) -> None:
        """Construye las filas y sus eventos a partir de datos del presentador."""
        for widget in self._filas.winfo_children():
            widget.destroy()
        self._controles_filas.clear()
        filas = filas_recibo(t.recibo, self._ui["moneda"]) if t.recibo else []
        for indice, fila in enumerate(filas):
            panel = ctk.CTkFrame(self._filas)
            panel.grid(row=indice, column=0, pady=4, sticky="ew")
            panel.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(panel, text=fila.nombre, anchor="w").grid(
                row=0, column=0, padx=8, sticky="w"
            )
            ctk.CTkLabel(panel, text=fila.precio_texto).grid(row=0, column=1, padx=8)
            ctk.CTkLabel(
                panel,
                text="REVISAR" if fila.estado == "revisar" else "",
                text_color=("#8a4500", "#ffbb66"),
            ).grid(row=1, column=0, sticky="w", padx=8)
            opciones = opciones_correccion(self.controlador.catalogo, t, fila.id_clase)
            menu = ctk.CTkOptionMenu(
                panel,
                values=[self._etiquetas[c] for c in opciones],
                command=lambda valor, actual=fila.id_clase: self.al_corregir(
                    actual, self._ids_opciones[valor]
                ),
            )
            menu.set("Cambiar a...")
            menu.grid(row=2, column=0, padx=8, pady=5, sticky="ew")
            quitar = ctk.CTkButton(
                panel,
                text="Quitar",
                width=75,
                command=lambda clase=fila.id_clase: self.al_quitar(clase),
            )
            quitar.grid(row=2, column=1, padx=8, pady=5)
            self._controles_filas.extend([menu, quitar])
        self._total.configure(text=texto_total(t.recibo, self._ui["moneda"]))

    def mostrar_alerta(self, mensaje: str) -> None:
        """Muestra un mensaje sin cerrar la ventana."""
        self._alerta.configure(text=mensaje)

    def al_escanear(self) -> None:
        """Elige una foto en Tk y envía lectura, captura y procesamiento al hilo."""
        if self._ocupada:
            return
        try:
            if self._desde_archivo:
                ruta = filedialog.askopenfilename(
                    parent=self,
                    title="Abrir foto del plato",
                    filetypes=[("Imágenes", "*.jpg *.jpeg *.png"), ("Todos", "*.*")],
                )
                if not ruta:
                    return
                tarea = partial(self.controlador.cargar_y_procesar, Path(ruta))
            else:
                tarea = self.controlador.capturar_y_procesar
            self._nueva_transaccion()

            if getattr(self, "_automatico", False):
                self._captura_auto.bloquear()
                self._origen_recibo = "archivo"

            self._iniciar_tarea(tarea)
        except Exception as error:  # noqa: BLE001 - frontera de eventos de Tk
            self.mostrar_alerta(str(error))

    def al_corregir(self, id_clase_actual: str, id_clase_nueva: str) -> None:
        """Solicita la corrección de todas las regiones del ítem."""
        if puede_editar(self.transaccion) and not self._ocupada:
            self._iniciar_tarea(
                partial(
                    self.controlador.corregir_item,
                    self.transaccion,
                    id_clase_actual,
                    id_clase_nueva,
                )
            )

    def al_quitar(self, id_clase: str) -> None:
        """Solicita quitar una clase del recibo."""
        if puede_editar(self.transaccion) and not self._ocupada:
            self._iniciar_tarea(
                partial(self.controlador.quitar_item, self.transaccion, id_clase)
            )

    def al_agregar(self, id_clase: str) -> None:
        """Solicita un alta manual en el recibo."""
        if puede_editar(self.transaccion) and not self._ocupada:
            self._iniciar_tarea(
                partial(self.controlador.agregar_item, self.transaccion, id_clase)
            )

    def al_confirmar(self) -> None:
        """Solicita guardar una transacción habilitada por el presentador."""
        if (
            self.transaccion is not None
            and not self._ocupada
            and puede_confirmar(self.transaccion, self._ui["exigir_revision"])
        ):
            self._iniciar_tarea(partial(self.controlador.confirmar, self.transaccion))

    def destroy(self) -> None:
        """Espera sin bloquear un trabajo pendiente antes de liberar la cámara."""
        if self._destruida:
            return
        if self._ocupada:
            self._cerrando = True
            self.withdraw()
            return
        self._destruida = True

        if self._consulta_camara is not None:
            self.after_cancel(self._consulta_camara)
            self._consulta_camara = None

        if self._consulta is not None:
            self.after_cancel(self._consulta)
        try:
            self.controlador.camara.liberar()
        finally:
            super().destroy()

    def _iniciar_tarea(self, tarea: Callable[[], Transaccion]) -> None:
        if self._ocupada or self._cerrando:
            return
        self._ocupada = True
        self._actualizar_controles()
        self.mostrar_alerta("Procesando...")
        self._hilo = Thread(
            target=_ejecutar_trabajo, args=(tarea, self._cola), daemon=True
        )
        try:
            self._hilo.start()
        except Exception as error:  # noqa: BLE001 - restablecer controles si falla el hilo
            self._ocupada = False
            self._actualizar_controles()
            self.mostrar_alerta(str(error))

    def _revisar_cola(self) -> None:
        self._consulta = None
        try:
            resultado = self._cola.get_nowait()
        except Empty:
            pass
        else:
            self._ocupada = False
            if self._cerrando:
                self.destroy()
                return
            try:
                if isinstance(resultado, Exception):
                    self.mostrar_alerta(
                        str(resultado) or "No se pudo completar la operación."
                    )
                else:
                    self.transaccion = resultado
                    self.mostrar_regiones(resultado)
                    self.mostrar_recibo(resultado)
                    self._tiempos.configure(text=texto_tiempos(resultado))
                    self.mostrar_alerta("\n".join(mensajes_alerta(resultado)))
            except Exception as error:  # noqa: BLE001 - frontera de actualización de Tk
                self.mostrar_alerta(str(error))
            self._actualizar_controles()
        self._consulta = self.after(50, self._revisar_cola)

    def _actualizar_controles(self) -> None:
        estado = "disabled" if self._ocupada else "normal"
        self._boton_escanear.configure(state=estado)
        self._boton_nueva.configure(state=estado)
        self._boton_marcas.configure(state=estado)
        self._boton_explicar.configure(state=estado)
        edicion = (
            "normal"
            if not self._ocupada and puede_editar(self.transaccion)
            else "disabled"
        )
        for control in self._controles_filas:
            control.configure(state=edicion)
        if not self._etiquetas:
            edicion = "disabled"
        self._opcion_agregar.configure(state=edicion)
        self._boton_agregar.configure(state=edicion)
        confirmar = (
            self.transaccion is not None
            and not self._ocupada
            and puede_confirmar(self.transaccion, self._ui["exigir_revision"])
        )
        self._boton_confirmar.configure(state="normal" if confirmar else "disabled")

    def _nueva_transaccion(self) -> None:
        if self._ocupada:
            return
        self.transaccion = None
        self._imagen_bgr = None
        self._con_marcas = True
        # Sustituir la imagen de Tk antes de soltar la referencia anterior.
        self._foto.configure(image=self._imagen_vacia, text="Abra una foto del plato")
        self._imagen_tk = self._imagen_vacia
        for contenedor in (self._filas, self._leyenda):
            for widget in contenedor.winfo_children():
                widget.destroy()
        self._controles_filas.clear()
        self._total.configure(text=texto_total(None, self._ui["moneda"]))
        self._tiempos.configure(text="")
        self.mostrar_alerta("")

        if self._automatico:
            self._origen_recibo = None
            self._pausada = False

            self._boton_pausa.configure(
                text="Pausar camara"
            )

            if self._captura_auto.fondo is not None:
                self._captura_auto.reiniciar()
                self.mostrar_alerta(
                    "Nueva transaccion. Camara lista para capturar."
                )
            else:
                self._inicio_camara = monotonic()
                self.mostrar_alerta(
                    "Mantenga vacia la superficie."
                )

        self._actualizar_controles()

    def _redibujar_foto(self) -> None:
        if self._imagen_bgr is None:
            return
        imagen = ajustar_a_caja(
            self._imagen_bgr,
            max(1, self._panel_imagen.winfo_width() - 12),
            max(1, self._panel_imagen.winfo_height() - 12),
        )
        rgb = Image.fromarray(imagen[:, :, ::-1])
        nueva = ctk.CTkImage(light_image=rgb, dark_image=rgb, size=rgb.size)
        self._foto.configure(image=nueva, text="")
        self._imagen_tk = nueva

    def _alternar_marcas(self) -> None:
        self._con_marcas = not self._con_marcas
        if self.transaccion is not None:
            self.mostrar_regiones(self.transaccion)

    def _agregar_seleccionado(self) -> None:
        id_clase = self._ids_opciones.get(self._opcion_agregar.get())
        if id_clase is not None:
            self.al_agregar(id_clase)

    def _mostrar_explicacion(self) -> None:
        if self.transaccion is None:
            return
        try:
            texto = texto_explicacion(self.transaccion, self.controlador.trazabilidad)
            ventana = ctk.CTkToplevel(self)
            ventana.title("¿Por qué se calculó este recibo?")
            ventana.geometry("700x480")
            cuadro = ctk.CTkTextbox(ventana, wrap="word")
            cuadro.pack(fill="both", expand=True, padx=10, pady=10)
            cuadro.insert("1.0", texto)
            cuadro.configure(state="disabled")
        except Exception as error:  # noqa: BLE001 - frontera del diálogo de explicación
            self.mostrar_alerta(str(error))


    def _alternar_camara(self):
        """Pausa o reanuda la captura automatica."""
        self._pausada = not self._pausada

        self._boton_pausa.configure(
            text="Reanudar camara"
            if self._pausada else "Pausar camara"
        )

        if self._pausada:
            self.mostrar_alerta("Camara pausada.")
        elif self.transaccion is None:
            self._captura_auto.reiniciar()
            self.mostrar_alerta("Camara activa. Coloque un plato.")

    def _calibrar_fondo(self):
        """Registra la superficie vacia como referencia."""
        if self._ocupada or self.transaccion is not None:
            return

        if self._ultimo_frame is None:
            self.mostrar_alerta("No hay imagen de la camara.")
            return

        self._captura_auto.calibrar(self._ultimo_frame)
        self.mostrar_alerta("Fondo calibrado. Coloque un plato.")

    def _actualizar_camara(self):
        """Actualiza la vista y dispara una captura cuando hay estabilidad."""
        self._consulta_camara = None

        if self._cerrando or self._destruida:
            return

        intervalo = max(
            33, int(self._ui.get("vista_previa_ms", 33))
        )

        try:
            if self._pausada:
                return

            frame = self.controlador.camara.leer_frame()
            self._ultimo_frame = frame

            if self._ocupada:
                return

            if self.transaccion is not None:

                if self.transaccion.estado == "confirmada":

                    if self._captura_auto.estado == "bloqueado":
                        self._captura_auto.esperar_retiro()

                    self._captura_auto.actualizar(frame)

                    if self._captura_auto.estado == "esperando":
                        self._nueva_transaccion()
                        self._captura_auto.calibrar(frame)

                elif self.transaccion.estado == "error":

                    if self._captura_auto.estado == "bloqueado":
                        self._captura_auto.esperar_retiro()

                    self._captura_auto.actualizar(frame)

                    if self._captura_auto.estado == "esperando":
                        self._nueva_transaccion()
                        self._captura_auto.calibrar(frame)

                return

            self.mostrar_previsualizacion(frame)

            if self._captura_auto.fondo is None:

                if monotonic() - self._inicio_camara >= 3.0:
                    self._captura_auto.calibrar(frame)
                    self.mostrar_alerta(
                        "Camara activa. Coloque un plato."
                    )

                return

            capturar = self._captura_auto.actualizar(frame)

            if capturar:
                self._origen_recibo = "camara"

                self._iniciar_tarea(
                    partial(
                        self.controlador.procesar,
                        frame.copy()
                    )
                )

            else:
                mensajes = {
                    "esperando": "Camara activa. Coloque un plato.",
                    "estabilizando": "Mantenga el plato inmovil...",
                    "esperando_retiro": "Retire el plato para continuar.",
                    "bloqueado": "Captura detenida. Inicie una nueva transaccion."
                }

                self.mostrar_alerta(
                    mensajes.get(self._captura_auto.estado, "")
                )

        except Exception as error:
            self.mostrar_alerta(
                f"Error de camara: {error}"
            )
            intervalo = 1000

        finally:
            if not self._cerrando and not self._destruida:
                self._consulta_camara = self.after(
                    intervalo,
                    self._actualizar_camara
                )
