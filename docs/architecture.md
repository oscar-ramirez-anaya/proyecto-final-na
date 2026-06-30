# Arquitectura — Proyecto Final CIL (MR4010.10)

Este documento describe la arquitectura técnica de la solución de Conditional
Imitation Learning (CIL): el flujo de datos, los modelos, los sensores y los
parámetros clave. Las instrucciones de ejecución están en el `README.md`.

---

## 1. Visión general del sistema

```
   MUNDO #1 (entrenamiento)                MUNDO #2 (evaluación)
   ┌────────────────────────┐             ┌──────────────────────────────┐
   │  Conducción manual      │             │  Conducción autónoma          │
   │  cil_data_collector     │             │  cil_autonomous               │
   │  - cámara -> PNG         │            │  - cámara -> modelo CIL        │
   │  - teclado: dirección    │   modelo   │  - comando por teclado         │
   │  - teclado: comando      │  .tflite   │  - capas de seguridad          │
   │  -> driving_log.csv      │ ─────────> │    (peatón, radar, evasión)    │
   └──────────┬─────────────┘             └──────────────┬───────────────┘
              │ dataset crudo (GitHub)                    │
              v                                           v
   ┌────────────────────────┐                    Tres rutas, comandos
   │  Google Colab           │                    distintos, sin U-turns
   │  cil_colab.ipynb         │
   │  - augmentation          │
   │  - entrena modelo CIL    │
   │  - exporta .keras/.tflite│
   └────────────────────────┘
```

---

## 2. Modelo CIL ramificado

```
 image (88x200x3)                            command (one-hot, 4)
      │                                            │
      v                                            │
 ┌──────────────────────────────────────────┐     │
 │ Bloque 1: Conv 5x5/s2 x32 -> Conv 3x3 x32  │    │
 │ Bloque 2: Conv 3x3/s2 x64 -> Conv 3x3 x64  │    │  cada conv:
 │ Bloque 3: Conv 3x3/s2 x128 -> Conv x128    │    │  BN + ReLU + Dropout(0.2)
 │ Bloque 4: Conv 3x3 x256 -> Conv 3x3 x256   │    │
 │ Flatten -> Dense(512) -> Dense(512) -> z    │   │
 └──────────────────┬───────────────────────┘     │
                    │ z                             │
     ┌──────────────┼───────────────┐              │
     v              v               v              │
  FOLLOW         LEFT/STRAIGHT     RIGHT            │
  256->256       256->256 ...      256->256         │
  Dense1(tanh)   Dense1(tanh)      Dense1(tanh)     │
     └──────────────┴───────────────┘              │
                    │ concat (N,4) * MAX_ANGLE      │
                    v                               │
                 [ * one-hot ] <────────────────────┘
                    │ reduce_sum
                    v
                steering (N,1)  en  [-0.5, 0.5] rad
```

Backbone fiel a Codevilla et al. (2018): 8 capas convolucionales en 4 bloques, con
BatchNormalization + ReLU + Dropout y reduccion espacial por stride 2. La seleccion
por máscara mantiene una única salida (MSE limpio) y un grafo exportable a TFLite. Se
omiten la rama de velocidad y las salidas de gas/freno del modelo original: la
velocidad no es entrada ni salida del modelo (la tarea indica que no se entrena).

---

## 3. Dispositivos del vehículo (BmwX5)

| Slot | Dispositivo | Nombre Webots | Uso |
|---|---|---|---|
| `sensorsSlotFront` | SickLms291 | `Sick LMS 291` | Distancia frontal (evasión, peatón) |
| `sensorsSlotTop` | Camera | `camera` | Imagen para CIL y Recognition de peatón |
| `sensorsSlotTop` | Display | `display_image` | Telemetría de a bordo |
| `sensorsSlotCenter` | GPS, Gyro | — | Odometría / rumbo |
| `sensorsSlotFront` | Radar | `radar` | Distancia de umbral al vehículo de adelante, junto al LiDAR (NUEVO) |
| `sensorsSlotCenter` | DistanceSensor x4 | `ds_right_front/mid/rear`, `ds_left` | Seguimiento de pared (evasión) |

---

## 4. Arbitraje de seguridad (controlador autónomo)

```
   leer cámara, LiDAR, radar, comando
              │
              v
   ¿peatón (Recognition + LiDAR, confirma SVM)?
        sí ──> dist < 8 m ? FRENO TOTAL : reducir a 10 km/h
        no
              v
   ¿radar: vehículo frontal < umbral+banda?
        sí ──> dist < 12 m ? DETENER : reducir velocidad proporcional
        no
              v
   ¿LiDAR: obstáculo estático < 14 m?
        sí ──> EVASIÓN (seguimiento de pared derecha, sobreescribe dirección)
        no
              v
   CIL: steering = modelo(imagen, comando)
              │
              v
   empuje anti-barandal (si ds_left < 3 m) -> saturar a +/-0.5 -> aplicar
```

---

## 5. Parámetros clave

| Parámetro | Valor | Significado |
|---|---|---|
| `IMG_H x IMG_W` | 88 x 200 | Entrada del modelo (estilo Codevilla) |
| Recorte de ROI | 40%–90% de la altura | Quita cielo y cofre |
| `MAX_ANGLE` | 0.5 rad | Límite del volante (entrenamiento e inferencia) |
| `COLLECT_SPEED` | 30 km/h | Velocidad constante de recolección |
| `CAPTURE_EVERY` | 3 pasos | Tasa de muestreo de imágenes |
| Augmentation | flip + brillo | Flip niega ángulo e intercambia LEFT<->RIGHT |
| `CRUISE_SPEED` | 25 km/h | Crucero autónomo en carril derecho |
| `PED_BRAKE_DIST` | 8.0 m | Freno total con peatón |
| `RADAR_DIST_UMBRAL` | 12.0 m | Distancia de umbral (declarada en el video) |
| `RADAR_SLOW_BAND` | 8.0 m | Banda de reducción proporcional sobre el umbral |
| `EVADE_APPROACH_DIST` | 14.0 m | Disparo de la evasión de obstáculo estático |

---

## 6. Preprocesamiento idéntico entrenamiento/inferencia

Para evitar el desfase dominio-entrenamiento, el recorte (40%–90% de altura),
la conversión BGR->RGB, el resize a 88x200 y la normalización `/255` son
idénticos en `train_cil.preprocess` y en `CILDriver.preprocess`. Cualquier cambio
debe replicarse en ambos lados.
