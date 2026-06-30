# Mundos de Webots — Proyecto Final CIL

Los mundos oficiales del proyecto (`MR4010_proyecto_final_2026.zip`) ya estan
**integrados y configurados** en esta carpeta:

| Archivo | Rol | Controlador asignado |
|---------|-----|----------------------|
| `city_traffic_2025_01.wbt` | Mundo #1 — entrenamiento (sin trafico) | `cil_data_collector` |
| `city_traffic_2025_02.wbt` | Mundo #2 — evaluacion (SUMO + 3 peatones + autos) | `cil_autonomous` |
| `city_traffic_2025_02_net/` | Red SUMO del Mundo #2 (la carga `SumoInterface`) | — |

> Para correrlos, abre el `.wbt` **desde este repositorio** (`navegacion_autonoma_final/`),
> de modo que Webots encuentre los controladores en `../controllers/`.

## Modificaciones ya aplicadas al `BmwX5`

Los mundos oficiales traian el vehiculo solo con Camara, GPS, Gyro y Display. Se
agregaron los sensores que pide el proyecto:

**Mundo #1** (`city_traffic_2025_01.wbt`):
- `controller "cil_data_collector"`.
- `Display` nombrado `display_image` (telemetria de a bordo).

**Mundo #2** (`city_traffic_2025_02.wbt`):
- `controller "cil_autonomous"`.
- `EXTERNPROTO` de `SickLms291` agregado a la cabecera.
- En `sensorsSlotFront`: **`SickLms291`** (LiDAR) + **`Radar`** (`name "radar"`,
  `maxRange 50`), montado **junto al LiDAR** como pide el enunciado.
- En la `Camera`: `recognition Recognition {}` (deteccion de peaton).
- En `sensorsSlotCenter`: `Display` nombrado `display_image` + **4 `DistanceSensor`**
  laterales (`ds_right_front/mid/rear`, `ds_left`) para la evasion (Act. 4.2).

Los peatones del Mundo #2 usan el proto `Pedestrian`, que define `model "pedestrian"`
y `recognitionColors`, por lo que el nodo Recognition los detecta y el filtro de
`cil_autonomous.detect_pedestrian_recognition` funciona sin cambios.

## Notas de operacion

- **Trafico SUMO:** el `SumoInterface` carga `city_traffic_2025_02_net/`. Se puede
  reducir el numero maximo de vehiculos en el Scene Tree si la PC lo requiere, pero
  **no por debajo de 30**.
- **Optional Rendering:** en `View > Optional Rendering` activa la visualizacion del
  LiDAR, el Radar y los rayos de los DistanceSensor para evidenciarlos en el video.
- **Distancia de umbral del radar:** `RADAR_DIST_UMBRAL = 12.0 m` (constante en
  `cil_autonomous.py`); es el valor que el video debe declarar.
- **Carril derecho y sin vueltas en U:** definir origen y destino de cada ruta sobre
  el lado derecho; reposicionar el `BmwX5` al origen antes de iniciar la grabacion.
