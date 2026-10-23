PS C:\Users\jacki\Desktop\cobro-vision> git grep -n "TODO(equipo)"
AGENTS.md:20:Si el código existente contradice la especificación, gana la especificación. Si la especificación es ambigua, elige la opción más simple, déjala anotada con `# TODO(equipo): <duda>` y continúa sin bloquearte.
AGENTS.md:53:10. Si falta información del equipo (menú real, precios, política de cobro), usa el valor de ejemplo, márcalo con `TODO(equipo)` y sigue.
AGENTS.md:62:- Decisiones que tomaste donde la especificación era ambigua (las mismas que dejaste como `TODO(equipo)`).
README.md:94:Decisiones de esta etapa marcadas como `TODO(equipo)`:
README.md:144:Decisiones de esta etapa marcadas como `TODO(equipo)`:
README.md:194:Decisiones de esta parte marcadas como `TODO(equipo)` en el código:
README.md:207:Decisiones de la parte B marcadas como `TODO(equipo)`:
README.md:298:<!-- # TODO(equipo): Confirmar el menú, los id_clase y los precios de ejemplo de data/catalogo_seed.csv. -->
README.md:307:Decisiones pendientes marcadas también como `TODO(equipo)` en el código:
config.yaml:6:  # TODO(equipo): Confirmar el menú, los id_clase y los precios de ejemplo de la semilla.
config.yaml:46:  # TODO(equipo): Calibrar este umbral provisional con 'evaluar clasificador'.
config.yaml:60:  # TODO(equipo): Confirmar la política real de cobro de la cafetería.
docs/ESPECIFICACION_TECNICA.md:31:| Menú y precios | Semilla de ejemplo (`TODO(equipo)`) | `data/catalogo_seed.csv` |
docs/ESPECIFICACION_TECNICA.md:488:`data/catalogo_seed.csv` (columnas `id_clase,nombre,categoria,precio_centavos,activo`). **Los precios son de ejemplo (`TODO(equipo)`); el equipo debe reemplazarlos.**
docs/ESPECIFICACION_TECNICA.md:677:- `CobroPorPlato`: si hay una proteína con precio en `por_plato.precios_por_proteina_centavos`, esa proteína cobra el precio del plato y el resto de ítems cobra 0; si no, se comporta como `CobroPorComponente`. `TODO(equipo)`: ajustar cuando la cafetería confirme su esquema.
docs/ESPECIFICACION_TECNICA.md:814:## 13. Pendientes que debe decidir el equipo (`TODO(equipo)`)
docs/PLAN_DE_ETAPAS.md:170:- `docs/PENDIENTES.md` con todos los `TODO(equipo)` encontrados en el código.
src/cobro_vision/adquisicion/camara.py:28:        # TODO(equipo): se incluyen subcarpetas para admitir data/raw/<SESION>.
src/cobro_vision/adquisicion/gestor_dataset.py:119:        # TODO(equipo): se conservan también filas de fotos retiradas para no perder
src/cobro_vision/adquisicion/gestor_dataset.py:142:        # TODO(equipo): se redondea la cantidad de platos de test hacia arriba,
