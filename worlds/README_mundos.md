# Mundos de Webots — Proyecto Final CIL

Los mundos base son los **oficiales** del proyecto, incluidos en el zip de Canvas
`MR4010_proyecto_final_2026.zip` (mundos `1` y `2`). El mundo `2` ya trae los tres
peatones, los cuatro vehiculos estacionados y el trafico generado por SUMO.

Pasos:

1. Descargar y descomprimir `MR4010_proyecto_final_2026.zip`.
2. Copiar el mundo de entrenamiento a `worlds/proyecto_final_mundo1.wbt` y el de
   evaluacion a `worlds/proyecto_final_mundo2.wbt`.
3. Asignar el controlador `cil_data_collector` al `BmwX5` del mundo 1 y
   `cil_autonomous` al `BmwX5` del mundo 2 (campo `controller`). En desarrollo se
   usa `controller "<extern>"` y se lanza el proceso Python por separado.
4. Aplicar las **modificaciones de sensores** descritas abajo al mundo 2 (y, si se
   desea evasion durante la recoleccion, tambien al mundo 1).

---

## Modificaciones al `BmwX5` (mundo 2)

Verificar que la `Camera` tenga `recognition Recognition {}` y que exista el
`SickLms291` con `name` por defecto `"Sick LMS 291"` en `sensorsSlotFront`
(ambos vienen en los mundos base de actividades previas).

Agregar el nodo **Radar** en `sensorsSlotFront [ ... ]` (junto al LiDAR, como pide
la tarea) y los **cuatro DistanceSensor laterales** en `sensorsSlotCenter [ ... ]`,
reutilizados de la Actividad 4.2 (habilitan la evasion de obstaculos estaticos). El
controlador los carga de forma defensiva: si alguno falta, esa capa de seguridad
simplemente no se activa.

El **Radar** se monta en el **slot delantero** del vehiculo, junto al LiDAR, tal como
sugiere el enunciado del proyecto.

```
  sensorsSlotFront [
    SickLms291 {
      translation 0.06 0 0
    }
    # --- COMPONENTE NUEVO: Radar para mantener distancia al vehiculo de adelante ---
    # Montado en el slot delantero junto al LiDAR (como pide la tarea). Los objetos
    # del mundo ya definen 'radarCrossSection', por lo que el radar los detecta sin
    # geometria adicional. El nombre "radar" coincide con cil_autonomous.py.
    Radar {
      name "radar"
      minRange 1
      maxRange 50
    }
  ]
  sensorsSlotCenter [
    GPS {
    }
    Gyro {
    }
    # --- Sensores laterales de un rayo (reutilizados de la Act. 4.2: evasion) ---
    DistanceSensor {
      translation 1.5 -1.05 0.30
      rotation 0 0 1 -1.5708
      name "ds_right_front"
      lookupTable [
        0 0 0
        5 5 0
      ]
      numberOfRays 1
      aperture 0.02
    }
    DistanceSensor {
      translation 0 -1.05 0.30
      rotation 0 0 1 -1.5708
      name "ds_right_mid"
      lookupTable [
        0 0 0
        5 5 0
      ]
      numberOfRays 1
      aperture 0.02
    }
    DistanceSensor {
      translation -1.5 -1.05 0.30
      rotation 0 0 1 -1.5708
      name "ds_right_rear"
      lookupTable [
        0 0 0
        5 5 0
      ]
      numberOfRays 1
      aperture 0.02
    }
    DistanceSensor {
      translation 0 1.05 0.30
      rotation 0 0 1 1.5708
      name "ds_left"
      lookupTable [
        0 0 0
        8 8 0
      ]
      numberOfRays 1
      aperture 0.05
    }
  ]
```

---

## Notas de operacion

- **Trafico SUMO:** se puede reducir el numero maximo de vehiculos en el Scene Tree
  (nodo `SumoInterface`) si la PC lo requiere, pero **no por debajo de 30**.
- **Optional Rendering:** habilitar en Webots la visualizacion de los rayos del
  LiDAR, del radar y de los sensores de distancia para que el video evidencie su
  operacion (menu `View > Optional Rendering`).
- **Peatones:** el controlador filtra los objetos de `Recognition` cuyo `model` sea
  `pedestrian`. Verificar en el Scene Tree que los peatones del mundo 2 tengan ese
  valor en su campo `model`; si difiere, ajustar la lista en
  `detect_pedestrian_recognition` (`cil_autonomous.py`).
- **Distancia de umbral del radar:** el valor declarado es
  `RADAR_DIST_UMBRAL = 12.0 m` (constante en `cil_autonomous.py`). Este es el valor
  que el video de evidencia debe mencionar; ajustarlo ahi si el equipo elige otro.
- **Carril derecho y sin vueltas en U:** definir origen y destino de cada una de las
  tres rutas sobre el lado derecho de conduccion; reposicionar el `BmwX5` al origen
  de cada ruta antes de iniciar la grabacion.
