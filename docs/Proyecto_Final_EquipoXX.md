# Proyecto Final — Conditional Imitation Learning (CIL)

**MR4010.10 Navegación Autónoma · Maestría en Inteligencia Artificial · Tecnológico de Monterrey**

> Documento base para exportar a DOCX/PDF con el nombre `Proyecto_Final_EquipoXX`
> (reemplazar `XX` por el número de equipo). Las secciones marcadas como
> *[PENDIENTE]* se completan tras la evaluación final en el Mundo #2.

---

## Portada

| | |
|---|---|
| **Institución** | Tecnológico de Monterrey |
| **Programa** | Maestría en Inteligencia Artificial |
| **Materia** | MR4010.10 — Navegación Autónoma |
| **Profesor** | Dr. David Antonio-Torres |
| **Actividad** | Proyecto Final — Conditional Imitation Learning |
| **Fecha** | Junio 2026 |

**Equipo:**

| Nombre | Matrícula |
|---|---|
| Antonio Olvera Donlucas | A01795617 |
| Carlos Monir Radovich Saad | A01797569 |
| Andres Roberto Osuna Gonzalez | A01796264 |
| Oscar Alberto Ramirez Anaya | A01795438 |

---

## 1. Objetivo

Aplicar los conceptos de Aprendizaje Máquina y Redes Neuronales Profundas a la
programación de un vehículo autónomo mediante **Behavioral Cloning con comandos de
navegación (Conditional Imitation Learning)**. Se recolecta un dataset conduciendo
manualmente en el Mundo #1, se entrena un modelo ramificado por comando y se evalúa
de forma autónoma en el Mundo #2 (con tráfico SUMO, peatones y vehículos estacionados).

## 2. Enfoque y fundamento

La arquitectura ramificada por comando sigue a **Codevilla et al. (2018)**: un
backbone convolucional común extrae características de la imagen y una rama
independiente por cada comando de navegación (FOLLOW, LEFT, STRAIGHT, RIGHT) predice
el ángulo de dirección. El comando activo selecciona qué rama produce la salida. El
backbone convolucional se inspira en **Bojarski et al. (2016)**. La velocidad se
mantiene constante y no forma parte del entrenamiento, conforme al enunciado.

## 3. Recolección del dataset (Mundo #1)

Controlador `cil_data_collector`:

- Conducción 100% manual (sin PID): flechas izquierda/derecha fijan el ángulo
  (paso 0.05 rad, límite ±0.5), flechas arriba/abajo la velocidad (tope 30 km/h).
- Comando de navegación latente con teclas 1–4, mostrado en el Display.
- Captura automática de imágenes (cada 3 pasos) y registro en `driving_log.csv`:
  `frame_idx, timestamp, image_name, steering_angle, command, speed`.
- Se recorre el mundo en ambos sentidos, cubriendo todas las rutas e incluyendo
  maniobras de recuperación (salir de la pista y volver) para robustez.

El dataset crudo se sube al repositorio público de GitHub `cil_dataset`. Tras el data
augmentation en Colab supera las **10 mil imágenes**.

## 4. Modelo y entrenamiento

- Entrada: imagen 66×200×3 (recorte 40%–90% de altura, RGB, normalizada) + comando one-hot.
- Backbone: 3 bloques Conv (32, 64, 64) + MaxPool + Dropout; Flatten; Dense(256).
- Cuatro ramas Dense(128)->Dense(1, tanh), escaladas a ±0.5 rad; selección por máscara.
- Pérdida MSE; Adam 1e-3; EarlyStopping + ReduceLROnPlateau.
- Data augmentation: flip horizontal (niega el ángulo e intercambia LEFT<->RIGHT) y
  jitter de brillo.
- Exportación a `.keras` y `.tflite` con verificación de paridad.

*[PENDIENTE: pegar las curvas de pérdida/MAE y la tabla de MAE por comando del notebook.]*

## 5. Evaluación autónoma (Mundo #2)

Controlador `cil_autonomous`, con arbitraje de seguridad por prioridad:

1. **Peatón + freno de emergencia:** nodo Recognition de la cámara (model `pedestrian`)
   en coordinación con el LiDAR; confirmación opcional con HOG+SVM. Freno total bajo 8 m.
2. **Distancia con radar:** el nodo Radar entrega la distancia al vehículo frontal más
   próximo. **Distancia de umbral declarada: `RADAR_DIST_UMBRAL = 12.0 m`.** Por debajo
   del umbral el vehículo se detiene; en la banda de aproximación reduce la velocidad
   de forma proporcional.
3. **Evasión de obstáculo estático:** seguimiento de pared derecha con sensores
   laterales (reutilizado de la Actividad 4.2).
4. **CIL (nominal):** el ángulo lo da el modelo según el comando del operador.

Se cubren **tres rutas**, cada una con un comando distinto dominante, favoreciendo el
carril derecho y sin vueltas en U. Al menos una ruta incluye evasión de obstáculo y al
menos una incluye detección de peatón con freno de emergencia.

*[PENDIENTE: describir las tres rutas elegidas (origen/destino) y pegar capturas.]*

## 6. Código de los controladores

### 6.1 Controlador de recolección (Mundo #1)
*[PEGAR el contenido completo de `controllers/cil_data_collector/cil_data_collector.py`
con sus comentarios.]*

### 6.2 Controlador autónomo (Mundo #2)
*[PEGAR el contenido completo de `controllers/cil_autonomous/cil_autonomous.py`
con sus comentarios.]*

### 6.3 Entrenamiento (notebook de Colab)
*[PEGAR las celdas de `cil_training/cil_colab.ipynb` y/o `train_cil.py`, destacando los
parámetros de la red y del entrenamiento que llevaron a la solución correcta.]*

## 7. Resultados y conclusiones

*[PENDIENTE: resumen de desempeño en las tres rutas, comportamiento de las capas de
seguridad y aprendizajes del equipo.]*

## 8. Video demostrativo

Enlace de YouTube (duración < 6 min): *[PENDIENTE]*

El video incluye la explicación de los parámetros de la red y del entrenamiento, la
evidencia de las tres rutas, la operación de los sensores (Optional Rendering), la
detección de peatón con freno y la **declaración explícita del valor de la distancia
de umbral del radar (12.0 m)**.

## 9. Declaración de uso de inteligencia artificial

Para la elaboración de este proyecto se utilizó **Claude Code**, bajo una metodología
de **spec-driven development y pruebas automatizadas**: la especificación de la tarea
se tradujo a un plan detallado y verificable, a partir del cual se generó y depuró el
código de los controladores, el modelo y el notebook, validándolo con pruebas
automatizadas (`tests/test_cil_inference.py`). La responsabilidad final sobre el
contenido recae en el equipo, que revisó y ajustó las soluciones para que se ajusten
estrictamente a lo solicitado.

## 10. Referencias

1. Codevilla, F., Müller, M., López, A., Koltun, V. y Dosovitskiy, A. (2018).
   *End-to-end Driving via Conditional Imitation Learning*. arXiv:1710.02410.
2. Bojarski, M. et al. (2016). *End to End Learning for Self-Driving Cars*. arXiv:1604.07316.
3. Ranjan, S. y Senthamilarasu, S. (2020). *Applied Deep Learning and Computer Vision
   for Self-Driving Cars*, cap. 10. Packt Publishing.
