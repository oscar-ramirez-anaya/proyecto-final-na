<h1 align="center">Proyecto Final — Conditional Imitation Learning (CIL)</h1>
<h3 align="center">MR4010.10 Navegacion Autonoma</h3>

<br>

<table align="center">
  <tr>
    <td><b>Institucion</b></td>
    <td>Instituto Tecnologico y de Estudios Superiores de Monterrey</td>
  </tr>
  <tr>
    <td><b>Programa</b></td>
    <td>Maestria en Inteligencia Artificial</td>
  </tr>
  <tr>
    <td><b>Materia</b></td>
    <td>MR4010.10 — Navegacion Autonoma</td>
  </tr>
  <tr>
    <td><b>Profesor</b></td>
    <td>Dr. David Antonio-Torres</td>
  </tr>
  <tr>
    <td><b>Fecha</b></td>
    <td>Junio 2026</td>
  </tr>
</table>

<h3 align="center">Equipo</h3>

<table align="center">
  <tr><th>Nombre</th><th>Matricula</th></tr>
  <tr><td>Antonio Olvera Donlucas</td><td>A01795617</td></tr>
  <tr><td>Carlos Monir Radovich Saad</td><td>A01797569</td></tr>
  <tr><td>Andres Roberto Osuna Gonzalez</td><td>A01796264</td></tr>
  <tr><td>Oscar Alberto Ramirez Anaya</td><td>A01795438</td></tr>
</table>

---

## Indice

1. [Introduccion](#1-introduccion)
2. [Descripcion del codigo base](#2-descripcion-del-codigo-base)
3. [Recoleccion del dataset (Mundo #1)](#3-recoleccion-del-dataset-mundo-1)
4. [Modelo CIL ramificado](#4-modelo-cil-ramificado)
5. [Data augmentation y entrenamiento](#5-data-augmentation-y-entrenamiento)
6. [Controlador autonomo y arbitraje de seguridad (Mundo #2)](#6-controlador-autonomo-y-arbitraje-de-seguridad-mundo-2)
7. [Deteccion de peaton y freno de emergencia](#7-deteccion-de-peaton-y-freno-de-emergencia)
8. [Distancia de umbral con radar](#8-distancia-de-umbral-con-radar)
9. [Evasion de obstaculos](#9-evasion-de-obstaculos)
10. [Mundos y sensores](#10-mundos-y-sensores)
11. [Pruebas automatizadas](#11-pruebas-automatizadas)
12. [Resultados](#12-resultados)
13. [Video demostrativo](#13-video-demostrativo)
14. [Declaracion de uso de inteligencia artificial](#14-declaracion-de-uso-de-inteligencia-artificial)
15. [Referencias](#15-referencias)
16. [Estructura del repositorio y ejecucion](#16-estructura-del-repositorio-y-ejecucion)

---

## 1. Introduccion

Este proyecto implementa una solucion de **Behavioral Cloning con comandos de
navegacion**, conocida como **Conditional Imitation Learning (CIL)**, para conducir
un vehiculo autonomo en Webots. La idea, propuesta por **Codevilla et al. (2018)**,
es que una sola red neuronal aprenda a conducir condicionada a un **comando de alto
nivel** (seguir carril, girar a la izquierda, seguir derecho, girar a la derecha),
de modo que el mismo modelo pueda tomar rutas distintas en una interseccion segun el
comando que reciba el operador.

El trabajo se organiza en tres fases:

1. **Recoleccion (Mundo #1):** se conduce el vehiculo **manualmente** a velocidad
   constante (<= 30 km/h) emitiendo comandos de navegacion por teclado, mientras un
   controlador captura imagenes de la camara de forma **automatica** y registra el
   angulo de direccion y el comando en un CSV.
2. **Entrenamiento (Google Colab):** con el dataset (> 10 mil imagenes tras data
   augmentation, alojado en GitHub) se entrena un **modelo CIL ramificado** y se
   exporta a TensorFlow Lite.
3. **Evaluacion (Mundo #2):** el modelo conduce de forma autonoma en un mundo con
   trafico SUMO, peatones y vehiculos estacionados, cubriendo tres rutas con
   comandos distintos y con **capas de seguridad** (evasion de obstaculos, deteccion
   de peaton con freno de emergencia y mantenimiento de distancia con radar).

La velocidad **no** forma parte del entrenamiento: se mantiene constante durante la
recoleccion y solo se modula en el Mundo #2 por las capas de seguridad.

---

## 2. Descripcion del codigo base

La solucion reutiliza componentes probados de las actividades previas del curso. La
siguiente tabla resume que se reutiliza y de donde proviene:

| Componente | Origen | Reuso en este proyecto |
|------------|--------|------------------------|
| `get_image` (camara BGRA -> Numpy) | Act. 2.1 / 3.1 / 4.2 | Ambos controladores |
| Control por teclado + anti-rebote | Act. 4.x (`sign_detector`) | Recoleccion y comando |
| Inferencia TFLite + respaldos de runtime | Act. 4.x (`sign_detector`) | Controlador autonomo |
| `process_lidar` (Sick LMS 291, zona central ±20) | Act. 4.2 (`evasion_obstaculos`) | Peaton y evasion |
| Deteccion por nodo Recognition | Act. 4.2 (`detect_bus_ahead`) | Deteccion de peaton |
| Deteccion HOG + SVM (`.joblib`) | Act. 3.1 (`autonomous_driver`) | Confirmacion de peaton |
| Freno de emergencia (`setBrakeIntensity`) | Act. 3.1 | Freno con peaton |
| Maquina de estados de seguimiento de pared | Act. 4.2 | Evasion de obstaculo |
| Backbone convolucional (Keras) | Act. 4.x (`train_gtsrb_cnn`) | Base del modelo CIL |

Los componentes **nuevos** de este proyecto son: (a) el **modo de recoleccion** con
CSV `(imagen, angulo, comando)`, (b) el **modelo CIL ramificado** y (c) la
integracion del nodo **Radar** para la distancia de umbral.

El vehiculo es un **BmwX5** (modelo Ackermann, API `vehicle.Driver`); el angulo del
volante se acota a **±0.5 rad** y la velocidad se expresa en km/h.

---

## 3. Recoleccion del dataset (Mundo #1)

El controlador [`controllers/cil_data_collector/cil_data_collector.py`](controllers/cil_data_collector/cil_data_collector.py)
es de **recoleccion asistida**: el vehiculo **se conduce solo siguiendo el carril**
con el pipeline de vision + PID reutilizado de la Act. 2.1 / 4.2, y el operador solo
**da el comando de navegacion** (un toque por interseccion). El controlador ejecuta el
giro y **etiqueta cada imagen automaticamente**. Asi el manejo recto (FOLLOW) es
automatico y suave, y el dataset sale mucho mas balanceado que conduciendo a mano.

### Controles de teclado

| Tecla | Accion |
|-------|--------|
| Q | Comando LEFT (un toque; el auto gira a la izquierda en la interseccion) |
| ↑ | Comando STRAIGHT (cruzar derecho) |
| E | Comando RIGHT (girar a la derecha) |
| ← / → | Correccion manual de la direccion (se suma al PID) |
| ↓ | Frena mientras se mantiene |
| G | Activa / desactiva la grabacion |

**FOLLOW es automatico:** no se presiona; el comando de giro regresa solo a FOLLOW
tras cruzar la interseccion (~4 s). La direccion se ralentiza durante los giros.

### Captura automatica y CSV

Cada `CAPTURE_EVERY = 3` pasos (no por pulsacion), si esta grabando y el auto se
mueve, se guarda la imagen y se anexa una fila al CSV:

```python
def save_sample(camera, frame_idx, steering, command, speed):
    image_name = f"{CMD_NAMES[command]}_{frame_idx:07d}.png"
    camera.saveImage(os.path.join(IMG_DIR, image_name), 100)
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            frame_idx, f"{time.time():.3f}", image_name,
            f"{steering:.5f}", command, f"{speed:.2f}",
        ])
```

Formato de `driving_log.csv`:

```
frame_idx, timestamp, image_name, steering_angle, command, speed
```

El nombre de archivo embebe el comando (`FOLLOW_0000123.png`) para inspeccionar el
balance del dataset de un vistazo. Recomendaciones de recoleccion: recorrer el mundo
en ambos sentidos cubriendo todas las rutas, incluir **maniobras de recuperacion**
(salir de la pista y volver) y mantener el carril derecho.

---

## 4. Modelo CIL ramificado

La arquitectura es fiel a **Codevilla et al. (2018)**: un **backbone convolucional**
comun extrae caracteristicas de la imagen y **una rama por comando** predice el
angulo de direccion. El comando activo selecciona, mediante una **mascara one-hot**,
que rama produce la salida.

```
 image (88x200x3)                                       command (one-hot, 4)
      │                                                       │
      v                                                       │
 ┌──────────────────────────────────────────────┐            │
 │ Bloque 1: Conv 5x5/s2 x32 -> Conv 3x3 x32       │           │
 │ Bloque 2: Conv 3x3/s2 x64 -> Conv 3x3 x64       │           │
 │ Bloque 3: Conv 3x3/s2 x128 -> Conv 3x3 x128     │  (cada conv: BN + ReLU + Dropout)
 │ Bloque 4: Conv 3x3 x256 -> Conv 3x3 x256        │           │
 │ Flatten -> Dense(512) -> Dense(512) -> z         │          │
 └───────────────────────┬──────────────────────┘            │
                          │ z                                  │
     ┌────────────────────┼────────────────────┐              │
     v          v                  v            v              │
  FOLLOW      LEFT             STRAIGHT        RIGHT            │
  256->256    256->256         256->256        256->256        │
  Dense1tanh  Dense1tanh       Dense1tanh      Dense1tanh      │
     └────────────────────┴────────────────────┘              │
                  concat (N,4) * MAX_ANGLE                      │
                          │                                     │
                          v                                     │
                      [ * one-hot ] <─────────────────────────-┘
                          │ reduce_sum
                          v
                    steering (N,1)  en  [-0.5, 0.5] rad
```

El cabezal de seleccion mantiene **una sola salida** (MSE limpio) y un grafo
exportable a TFLite:

```python
todas = layers.Concatenate(name="branches")(ramas)          # (N, 4)
todas = layers.Lambda(lambda t: t * MAX_ANGLE)(todas)       # escala a rango fisico
seleccion = layers.Multiply()([todas, cmd_in])              # (N,4) * one-hot
steering  = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1, keepdims=True))(seleccion)
```

Diferencias respecto al modelo original de Codevilla: se **omite** la rama de
velocidad y las salidas de gas/freno (la tarea indica que la velocidad no se
entrena), por lo que cada rama produce **solo el angulo de direccion**.

---

## 5. Data augmentation y entrenamiento

El data augmentation es especifico de Behavioral Cloning. La tecnica clave es el
**flip horizontal**, que **niega el angulo** e **intercambia el comando LEFT<->RIGHT**
(FOLLOW y STRAIGHT no cambian). Esto duplica el dataset y lo balancea entre giros:

```python
X_flip = X[:, :, ::-1, :].copy()                 # espejo en el eje del ancho
y_flip = -y                                       # angulo negado
C_flip = np.array([CMD_FLIP[int(c)] for c in C])  # LEFT<->RIGHT
```

Se agrega ademas **jitter de brillo** para generalizar entre el Mundo #1 y el #2. El
notebook valida que el dataset final supere las **10 mil imagenes**.

| Hiperparametro | Valor |
|----------------|-------|
| Entrada | 88 x 200 x 3 (recorte 40%-90% de altura, RGB, /255) |
| Optimizador | Adam, lr = 1e-3 |
| Perdida / metrica | MSE / MAE de steering |
| Callbacks | EarlyStopping (patience 8), ReduceLROnPlateau |
| Batch / epocas | 64 / 40 (con early stopping) |
| Exportacion | `.keras` + `.tflite` con verificacion de paridad |

El entrenamiento se ejecuta en [`cil_training/cil_colab.ipynb`](cil_training/cil_colab.ipynb),
que clona el dataset desde GitHub (`!git clone`), entrena, evalua el **MAE por
comando** y descarga `cil_model.tflite`. La logica vive en
[`cil_training/train_cil.py`](cil_training/train_cil.py), reutilizada por el notebook.

---

## 6. Controlador autonomo y arbitraje de seguridad (Mundo #2)

El controlador [`controllers/cil_autonomous/cil_autonomous.py`](controllers/cil_autonomous/cil_autonomous.py)
combina la inferencia del modelo CIL con tres capas de seguridad, arbitradas **por
prioridad** (seguridad > CIL):

```
   leer camara, LiDAR, radar, comando del operador
              │
              v
   (1) ¿peaton (Recognition + LiDAR, confirma SVM)?
        si ──> dist < 8 m ? FRENO TOTAL : reducir a 10 km/h
        no
              v
   (2) ¿radar: vehiculo frontal < umbral + banda?
        si ──> dist < 12 m ? DETENER : reducir velocidad proporcional
        no
              v
   (3) ¿LiDAR: obstaculo estatico < 14 m?
        si ──> EVASION (seguimiento de pared, sobreescribe direccion)
        no
              v
   (4) CIL: steering = modelo(imagen, comando)
              │
              v
   empuje anti-barandal (si ds_left < 3 m) -> saturar ±0.5 -> aplicar
```

La inferencia condicionada al comando se realiza con la clase `CILDriver`, que
preprocesa la imagen **igual que en el entrenamiento** y alimenta las dos entradas
(imagen + comando one-hot):

```python
def predict(self, bgra, command):
    x = np.expand_dims(self.preprocess(bgra), axis=0)        # (1,88,200,3)
    onehot = np.zeros((1, NUM_COMMANDS), dtype=np.float32)
    onehot[0, command] = 1.0
    self.interpreter.set_tensor(self.img_in["index"], x)
    self.interpreter.set_tensor(self.cmd_in["index"], onehot)
    self.interpreter.invoke()
    steering = float(self.interpreter.get_tensor(self.out["index"]).ravel()[0])
    return float(np.clip(steering, -MAX_ANGLE, MAX_ANGLE))
```

El operador entrega el comando de navegacion durante la ruta con las teclas `1`-`4`
(latched). La carga de sensores es **defensiva**: si el mundo no incluye radar o los
sensores laterales, esa capa simplemente no se activa y el resto sigue operando.

---

## 7. Deteccion de peaton y freno de emergencia

La deteccion la realiza el **nodo Recognition de la camara** (filtrando `model`
`pedestrian`) **en coordinacion con el LiDAR**, como pide la tarea. Opcionalmente se
confirma con HOG + SVM (modelo reutilizado de la Act. 3.1):

```python
def detect_pedestrian_recognition(camera):
    objects = camera.getRecognitionObjects()
    cam_cx = camera.getWidth() / 2.0
    for obj in objects:
        if obj.getModel() in ("pedestrian", "Pedestrian", "human"):
            pos = obj.getPositionOnImage()
            if abs(pos[0] - cam_cx) < cam_cx * 0.8:   # tercio central
                return True
    return False
```

Si hay peaton y el LiDAR reporta una distancia frontal menor a `PED_BRAKE_DIST = 8 m`,
se aplica **freno total** (`setBrakeIntensity(1.0)` + `setCruisingSpeed(0)`); si esta
mas lejos, se reduce a 10 km/h por precaucion.

---

## 8. Distancia de umbral con radar

Componente **nuevo** del proyecto. El nodo **Radar** de Webots entrega distancia,
velocidad relativa y azimut por objetivo, lo que lo hace ideal para mantener la
distancia al vehiculo de adelante:

```python
def read_radar_nearest(radar):
    targets = radar.getTargets()
    mejor = None
    for t in targets:
        if abs(t.getAzimuth()) <= RADAR_AZIMUTH_MAX:   # solo casi al frente
            if mejor is None or t.getDistance() < mejor:
                mejor = t.getDistance()
    return mejor
```

La **distancia de umbral declarada es `RADAR_DIST_UMBRAL = 12.0 m`** (este es el
valor que el video de evidencia debe mencionar). El comportamiento es:

| Distancia al vehiculo frontal | Accion |
|-------------------------------|--------|
| > 20 m (umbral + banda) | Velocidad crucero normal |
| 12 m – 20 m | Reduccion **proporcional** de la velocidad |
| < 12 m (umbral) | **Detenerse** (freno total) |

---

## 9. Evasion de obstaculos

Para los obstaculos estaticos del Mundo #2 se reutiliza el **seguimiento de pared
derecha** de la Actividad 4.2. Cuando el LiDAR detecta un objeto en el carril (y no
es un peaton ni un vehiculo gestionado por el radar) dentro de `EVADE_APPROACH_DIST = 14 m`,
se activa la maniobra, que **sobreescribe** la direccion del modelo:

```python
def _wall_following(ds_left, ds_right_front, ds_right_rear, steps):
    rf = ds_right_front.getValue(); rr = ds_right_rear.getValue()
    # Reincorporacion: costado derecho libre tras el minimo de pasos.
    if steps > MIN_EVADE_STEPS and rf >= WALL_CLEAR_DIST and rr >= WALL_CLEAR_DIST:
        return 0.0, EVADE_SPEED, False
    steer = EVADE_STEER                       # girar a la izquierda para rodear
    if rf >= WALL_CLEAR_DIST:
        steer = EVADE_STEER * 0.4             # ya rebaso el frente: enderezar
    return steer, EVADE_SPEED, True
```

Un **empuje anti-barandal** (activo en todos los estados) corrige hacia la derecha si
`ds_left` detecta el barandal a menos de 3 m.

---

## 10. Mundos y sensores

Los mundos oficiales del zip de Canvas ya estan **integrados y configurados** en
[`worlds/`](worlds/): `city_traffic_2025_01.wbt` (Mundo #1, entrenamiento) y
`city_traffic_2025_02.wbt` (Mundo #2, con trafico SUMO, tres peatones y vehiculos
estacionados). Al `BmwX5` se le agregaron el **LiDAR + Radar** (slot delantero), el
nodo **Recognition** en la camara y los **sensores laterales** de la Act. 4.2; los
controladores ya quedan asignados. El detalle esta en
[`worlds/README_mundos.md`](worlds/README_mundos.md).

| Slot | Dispositivo | Nombre Webots | Uso |
|------|-------------|---------------|-----|
| `sensorsSlotFront` | SickLms291 | `Sick LMS 291` | Distancia frontal (peaton, evasion) |
| `sensorsSlotTop` | Camera (+ Recognition) | `camera` | Imagen CIL + deteccion de peaton |
| `sensorsSlotTop` | Display | `display_image` | Telemetria de a bordo |
| `sensorsSlotFront` | Radar | `radar` | Distancia de umbral, junto al LiDAR (NUEVO) |
| `sensorsSlotCenter` | DistanceSensor x4 | `ds_right_*`, `ds_left` | Seguimiento de pared (evasion) |

Notas de operacion: reducir el maximo de vehiculos SUMO en el Scene Tree **sin bajar
de 30**; habilitar **Optional Rendering** para visualizar los sensores en el video;
favorecer el carril derecho y no realizar vueltas en U.

---

## 11. Pruebas automatizadas

El archivo [`tests/test_cil_inference.py`](tests/test_cil_inference.py) valida la
inferencia sin necesidad de Webots:

1. **Seleccion por mascara one-hot** (NumPy puro, corre en cualquier entorno):
   verifica que la salida del modelo sea exactamente la de la rama del comando activo.
2. **Inferencia TFLite** (si hay TensorFlow): comprueba que el modelo responde con un
   angulo finito en `[-0.5, 0.5]` para cada comando y que las cuatro ramas
   **discriminan** (no devuelven el mismo angulo).

```
$ python tests/test_cil_inference.py
[PASS] Seleccion por mascara one-hot correcta
[PASS] Inferencia TFLite OK; angulos por comando: [...]
```

---

## 12. Resultados

> Esta seccion se completa tras la evaluacion final en el Mundo #2. Las evidencias
> (curvas de entrenamiento, MAE por comando y capturas de las tres rutas) se guardan
> en [`screenshots/`](screenshots/) y se integran al reporte
> [`docs/Proyecto_Final_EquipoXX.md`](docs/Proyecto_Final_EquipoXX.md).

Checklist de evaluacion (paso 6 de la tarea):

- [ ] Tres rutas con comandos distintos (FOLLOW / LEFT / STRAIGHT / RIGHT).
- [ ] Al menos una ruta con evasion de obstaculo estatico.
- [ ] Al menos una ruta con deteccion de peaton (Recognition + LiDAR) y freno de emergencia.
- [ ] Distancia de umbral con radar declarada (12 m) y demostrada.
- [ ] Carril derecho respetado, sin vueltas en U, SUMO >= 30 vehiculos.

---

## 13. Video demostrativo

[Enlace al video en YouTube](#) *(pendiente; duracion < 6 min)*

El video explica los parametros de la red y del entrenamiento, muestra las tres
rutas, la operacion de los sensores (Optional Rendering), la deteccion de peaton con
freno y **declara el valor de la distancia de umbral del radar (12 m)**.

---

## 14. Declaracion de uso de inteligencia artificial

Para el desarrollo de este proyecto se utilizo **Claude Code**, bajo una metodologia
de **spec-driven development y pruebas automatizadas**: la especificacion de la tarea
se tradujo a un plan detallado y verificable, a partir del cual se genero y depuro el
codigo de los controladores, el modelo y el notebook, validandolo con las pruebas
automatizadas (`tests/test_cil_inference.py`). Todo el codigo fue revisado por el
equipo; la responsabilidad final del contenido recae en los autores.

---

## 15. Referencias

1. Codevilla, F., Muller, M., Lopez, A., Koltun, V. y Dosovitskiy, A. (2018).
   *End-to-end Driving via Conditional Imitation Learning*. arXiv:1710.02410.
   https://arxiv.org/pdf/1710.02410
2. Codevilla, F. *Imitation Learning (CARLA)* — repositorio de referencia.
   https://github.com/carla-simulator/imitation-learning
3. Bojarski, M. et al. (2016). *End to End Learning for Self-Driving Cars*.
   arXiv:1604.07316. https://arxiv.org/abs/1604.07316
4. Ranjan, S. y Senthamilarasu, S. (2020). *Applied Deep Learning and Computer Vision
   for Self-Driving Cars*, cap. 10. Packt Publishing.
5. Cyberbotics Ltd. *Webots Reference Manual — Radar / Camera Recognition / Lidar*.
   https://cyberbotics.com/doc/reference/radar

---

## 16. Estructura del repositorio y ejecucion

```
navegacion_autonoma_final/
├── README.md                          # Este reporte
├── LICENSE                            # Apache 2.0
├── .gitignore
├── docs/
│   ├── architecture.md                # Diagramas, tablas de sensores y parametros
│   └── Proyecto_Final_EquipoXX.md     # Borrador del reporte de entrega
├── cil_training/
│   ├── cil_colab.ipynb                # Notebook de entrenamiento (clona dataset, entrena, exporta)
│   ├── train_cil.py                   # Modelo CIL ramificado + augmentation + exportacion
│   ├── requirements.txt
│   └── model/                         # cil_model.keras / cil_model.tflite (tras entrenar)
├── controllers/
│   ├── cil_data_collector/            # Mundo #1: recoleccion manual + CSV
│   │   ├── cil_data_collector.py
│   │   └── requirements.txt
│   └── cil_autonomous/                # Mundo #2: inferencia CIL + capas de seguridad
│       ├── cil_autonomous.py
│       ├── svm_pedestrian_model.joblib
│       ├── model/                     # cil_model.tflite (copiar tras entrenar)
│       └── requirements.txt
├── worlds/
│   ├── city_traffic_2025_01.wbt       # Mundo #1 (entrenamiento) — controlador cil_data_collector
│   ├── city_traffic_2025_02.wbt       # Mundo #2 (evaluacion) — controlador cil_autonomous
│   ├── city_traffic_2025_02_net/      # Red SUMO del Mundo #2
│   └── README_mundos.md               # Detalle de los sensores integrados al BmwX5
├── tests/
│   └── test_cil_inference.py          # Pruebas de inferencia
└── screenshots/                       # Evidencias (entrenamiento, rutas, sensores)
```

### Ejecucion

Los mundos oficiales ya estan integrados en `worlds/` (sensores y controladores
asignados). Pasos:

1. **Runtime (solo si hace falta):** si el Python de Webots no tiene las dependencias
   de `requirements.txt`, edita `controllers/<nombre>/runtime.ini` y apunta
   `PYTHONPATH` a tu entorno:

   ```ini
   [environment variables]
   PYTHONPATH = /ruta/a/tu/env/lib/python3.x/site-packages
   ```

2. **Recoleccion (Mundo #1):** abrir `worlds/city_traffic_2025_01.wbt` desde este
   repositorio y pulsar Play. El controlador `cil_data_collector` ya esta asignado;
   conducir genera `controllers/cil_data_collector/dataset/IMG/` + `driving_log.csv`.
3. **Dataset:** subir `dataset/` al repo `cil_dataset` en GitHub.
4. **Entrenamiento:** ejecutar `cil_training/cil_colab.ipynb` en Google Colab y
   copiar `cil_model.tflite` a `controllers/cil_autonomous/model/`.
5. **Evaluacion (Mundo #2):** abrir `worlds/city_traffic_2025_02.wbt`, pulsar Play y
   dar comandos por teclado (`1`-`4`) para recorrer las tres rutas.
6. **Pruebas:** `python tests/test_cil_inference.py`.

<br>
<p align="center"><i>Instituto Tecnologico y de Estudios Superiores de Monterrey
— Maestria en Inteligencia Artificial — Junio 2026</i></p>
