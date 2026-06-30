<h1 align="center">Proyecto Final — Conditional Imitation Learning (CIL)</h1>
<h3 align="center">MR4010.10 Navegacion Autonoma</h3>
<br>
<table align="center">
  <tr><td><b>Institucion</b></td><td>Tecnologico de Monterrey</td></tr>
  <tr><td><b>Programa</b></td>   <td>Maestria en Inteligencia Artificial</td></tr>
  <tr><td><b>Materia</b></td>    <td>MR4010.10 — Navegacion Autonoma</td></tr>
  <tr><td><b>Profesor</b></td>   <td>Dr. David Antonio-Torres</td></tr>
  <tr><td><b>Fecha</b></td>      <td>Junio 2026</td></tr>
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
3. [Dataset utilizado](#3-dataset-utilizado)
4. [Modelo CIL ramificado](#4-modelo-cil-ramificado)
5. [Entrenamiento y evaluacion](#5-entrenamiento-y-evaluacion)
6. [Controlador autonomo y capas de seguridad](#6-controlador-autonomo-y-capas-de-seguridad)
7. [Integracion en Webots](#7-integracion-en-webots)
8. [Resultados](#8-resultados)
9. [Conclusiones](#9-conclusiones)
10. [Video demostrativo](#10-video-demostrativo)
11. [Referencias](#11-referencias)
12. [Estructura del repositorio y ejecucion](#12-estructura-del-repositorio-y-ejecucion)

---

## 1. Introduccion

Este proyecto implementa una solucion de **Behavioral Cloning con comandos de
navegacion**, conocida como **Conditional Imitation Learning (CIL)**, para conducir
un vehiculo autonomo en el simulador Webots. El conductor humano recolecta un dataset
en el Mundo #1 (conduccion manual a velocidad constante) mientras emite comandos de
navegacion (seguir, girar a la izquierda, seguir derecho, girar a la derecha). Con ese
dataset se entrena una red neuronal ramificada que aprende a predecir el angulo de
direccion condicionado al comando, y el modelo se evalua de forma autonoma en el
Mundo #2, que incluye trafico de SUMO, peatones y vehiculos estacionados.

El enfoque sigue el articulo de Codevilla et al. (2018) para la arquitectura
ramificada por comando, y el backbone convolucional se inspira en Bojarski et al.
(2016).

## 2. Descripcion del codigo base

El proyecto reutiliza componentes probados de las actividades previas del curso:

- **Pipeline de camara y seguimiento de carril** (Act. 2.1): captura BGRA, Canny, ROI
  trapezoidal y Hough.
- **LiDAR Sick LMS 291** (Act. 3.1 / 4.2): lectura del range image 1D en la zona central.
- **Deteccion de peaton** (Act. 3.1): HOG + SVM; y nodo Recognition de la camara (Act. 4.2).
- **Evasion de obstaculos** (Act. 4.2): seguimiento de pared derecha con sensores laterales.
- **Inferencia TFLite** (Act. 4.x): interprete con respaldos de runtime.

El componente **nuevo** del proyecto es la lectura del nodo **Radar** para mantener
una distancia de umbral al vehiculo de adelante, junto con el **modo de recoleccion
con CSV** y el **modelo CIL ramificado**.

## 3. Dataset utilizado

El controlador `cil_data_collector` (Mundo #1) captura imagenes de la camara de forma
automatica y registra una fila por imagen en `driving_log.csv`:

```
frame_idx, timestamp, image_name, steering_angle, command, speed
```

El dataset crudo se sube a un repositorio publico de GitHub
(`cil_dataset`) para que el notebook de Colab lo clone con `!git clone`. Tras el data
augmentation (flip horizontal con negacion del angulo e intercambio LEFT<->RIGHT, mas
jitter de brillo) el dataset supera las **10 mil imagenes** y queda balanceado entre
ambos sentidos de conduccion.

## 4. Modelo CIL ramificado

Arquitectura (Keras funcional, dos entradas y una salida):

```
imagen (66,200,3) --> Conv32 x2 -> Pool -> Dropout
                      Conv64 x2 -> Pool -> Dropout
                      Conv64 x2 -> Pool -> Dropout
                      Flatten -> Dense(256) -> z
        z --> Rama FOLLOW   : Dense(128) -> Dense(1, tanh) \
        z --> Rama LEFT     : Dense(128) -> Dense(1, tanh)  \  apila (N,4) -> *MAX_ANGLE
        z --> Rama STRAIGHT : Dense(128) -> Dense(1, tanh)  /  * comando one-hot
        z --> Rama RIGHT    : Dense(128) -> Dense(1, tanh) /   -> suma -> steering (N,1)
```

El comando one-hot selecciona, por mascara, la rama activa. La velocidad **no** entra
al modelo (se mantiene constante y no se entrena). La perdida es MSE sobre el angulo
de la rama seleccionada.

## 5. Entrenamiento y evaluacion

Se entrena en Google Colab (`cil_training/cil_colab.ipynb`) con Adam (1e-3),
EarlyStopping y ReduceLROnPlateau. La evaluacion reporta el MAE de steering global y
desglosado por comando. La exportacion genera `.keras` y `.tflite` con verificacion de
paridad Keras vs TFLite.

## 6. Controlador autonomo y capas de seguridad

`cil_autonomous` (Mundo #2) arbitra por prioridad:

```
1. PEATON (Recognition + LiDAR, confirma SVM) -> freno total
2. RADAR  (vehiculo mas proximo < umbral)     -> detener / reducir velocidad
3. EVASION (obstaculo estatico, LiDAR + sensores laterales) -> sobreescribe direccion
4. CIL (nominal) -> angulo segun el comando del operador
```

La distancia de umbral del radar declarada es **12.0 m** (`RADAR_DIST_UMBRAL`).

## 7. Integracion en Webots

Ver [`worlds/README_mundos.md`](worlds/README_mundos.md) para las modificaciones del
`BmwX5` (nodo Radar y sensores laterales) sobre los mundos oficiales del zip de Canvas.

## 8. Resultados

Pendiente: completar con las metricas finales de entrenamiento (MAE por comando),
capturas de las tres rutas del Mundo #2 y evidencia de las capas de seguridad. Las
imagenes se guardan en `screenshots/`.

## 9. Conclusiones

Pendiente: redactar tras la evaluacion final.

## 10. Video demostrativo

Pendiente: enlace de YouTube (duracion < 6 min).

## 11. Referencias

1. Codevilla, F. et al. (2018). *End-to-end Driving via Conditional Imitation Learning*. arXiv:1710.02410.
2. Bojarski, M. et al. (2016). *End to End Learning for Self-Driving Cars*. arXiv:1604.07316.
3. Ranjan, S. y Senthamilarasu, S. (2020). *Applied Deep Learning and Computer Vision for Self-Driving Cars*, cap. 10. Packt.

## 12. Estructura del repositorio y ejecucion

```
navegacion_autonoma_final/
├── controllers/
│   ├── cil_data_collector/   # Mundo #1: recoleccion manual + CSV
│   └── cil_autonomous/       # Mundo #2: inferencia CIL + seguridad
├── cil_training/             # notebook Colab + train_cil.py + modelo
├── worlds/                   # mundos oficiales + nodo Radar (ver README_mundos.md)
├── docs/                     # architecture.md + reporte
├── tests/                    # pruebas automatizadas de inferencia
└── screenshots/
```

Ejecucion resumida:

```
# 1. Recolectar dataset (Webots, Mundo #1) con el controlador cil_data_collector
# 2. Subir dataset/ al repo cil_dataset en GitHub
# 3. Entrenar en Colab (cil_colab.ipynb) y descargar cil_model.tflite
# 4. Copiar cil_model.tflite a controllers/cil_autonomous/model/
# 5. Evaluar (Webots, Mundo #2) con el controlador cil_autonomous
# 6. Pruebas:  python tests/test_cil_inference.py
```

---

<p align="center"><i>Instituto Tecnologico y de Estudios Superiores de Monterrey — Maestria en Inteligencia Artificial — Junio 2026</i></p>
