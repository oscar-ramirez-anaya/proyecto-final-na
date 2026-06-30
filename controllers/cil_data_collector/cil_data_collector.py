"""
===============================================================================
  Proyecto Final — Conditional Imitation Learning (CIL)  ·  MR4010.10
  Controlador de RECOLECCION de datos (Mundo #1)
===============================================================================

  Proposito
  ---------
  Conducir el vehiculo MANUALMENTE en el Mundo #1 mientras se registra, de forma
  automatica, el dataset necesario para entrenar un modelo de Behavioral Cloning
  con comandos de navegacion (Conditional Imitation Learning, Codevilla et al.,
  arXiv:1710.02410).

  A diferencia del controlador de la Actividad 2.1, aqui NO hay PID: la direccion
  la fija por completo el conductor humano con el teclado. El controlador unicamente:
    1. Aplica al vehiculo el angulo de direccion y la velocidad indicados por teclado.
    2. Mantiene un "comando de navegacion" latente (latched) que el conductor cambia
       en cada interseccion (seguir/izquierda/recto/derecha).
    3. Captura imagenes de la camara de forma AUTOMATICA (cada CAPTURE_EVERY pasos,
       no por pulsacion) y anexa una fila al CSV con (imagen, angulo, comando).

  El pipeline de captura de camara (get_image) se reutiliza sin cambios de las
  actividades previas (Act. 2.1 / 3.1 / 4.2).

  Controles de teclado
  --------------------
    Flecha IZQUIERDA / DERECHA : ajustan el angulo de direccion (lo que se entrena).
    Flecha ARRIBA   / ABAJO    : ajustan la velocidad crucero (acotada a <=30 km/h).
    R                          : enderezan el volante (angulo = 0).
    1 / 2 / 3 / 4              : fijan el comando de navegacion latente:
                                   1 = FOLLOW   (seguir carril)
                                   2 = LEFT     (girar a la izquierda en la interseccion)
                                   3 = STRAIGHT (seguir derecho en la interseccion)
                                   4 = RIGHT    (girar a la derecha en la interseccion)
    G                          : activa / desactiva la grabacion (toggle).
    S                          : detiene el vehiculo (velocidad 0).

  Salida (dataset)
  ----------------
    dataset/IMG/<comando>_<frame>.png   imagenes capturadas
    dataset/driving_log.csv             una fila por imagen:
        frame_idx, timestamp, image_name, steering_angle, command, speed

  Nota: la velocidad se mantiene constante y baja (<=30 km/h) durante la
  recoleccion y NO forma parte del entrenamiento (solo se registra como
  referencia), tal como pide el enunciado del proyecto.

  Equipo:
      Antonio Olvera Donlucas          A01795617
      Carlos Monir Radovich Saad       A01797569
      Andres Roberto Osuna Gonzalez    A01796264
      Oscar Alberto Ramirez Anaya      A01795438

  Institucion:
      Instituto Tecnologico y de Estudios Superiores de Monterrey
      Maestria en Inteligencia Artificial  ·  MR4010.10 Navegacion Autonoma
===============================================================================
"""

import os
import csv
import time
import numpy as np

# Imports de Webots
from controller import Display, Keyboard
from vehicle import Driver


# ============================================================
# 1. CONSTANTES
# ============================================================

# --- Direccion y velocidad (limites reutilizados de la Act. 2.1 / 3.1) ---
MAX_ANGLE   = 0.5     # rad — angulo maximo del volante (giros centrales abruptos lo requieren)
ANGLE_INCR  = 0.05    # rad por pulsacion de flecha izquierda/derecha
COLLECT_SPEED = 30.0  # km/h — velocidad de recoleccion (constante, NO se entrena)
MAX_SPEED   = 30.0    # km/h — tope duro: el enunciado exige <=30 km/h al recolectar
SPEED_INCR  = 5.0     # km/h por pulsacion de flecha arriba/abajo

# --- Comandos de navegacion (Conditional Imitation Learning) ---
CMD_FOLLOW   = 0      # seguir carril
CMD_LEFT     = 1      # girar a la izquierda en la siguiente interseccion
CMD_STRAIGHT = 2      # seguir derecho en la interseccion
CMD_RIGHT    = 3      # girar a la derecha en la siguiente interseccion
CMD_NAMES = {CMD_FOLLOW: "FOLLOW", CMD_LEFT: "LEFT",
             CMD_STRAIGHT: "STRAIGHT", CMD_RIGHT: "RIGHT"}

# --- Captura del dataset ---
CAPTURE_EVERY = 3     # guardar 1 imagen cada N pasos de simulacion (controla la tasa)
DEBOUNCE_TIME = 0.10  # s — anti-rebote para teclas de un solo disparo (R, G, S, comandos)

# --- Rutas de salida (relativas al controlador) ---
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
IMG_DIR     = os.path.join(DATASET_DIR, "IMG")
CSV_PATH    = os.path.join(DATASET_DIR, "driving_log.csv")
CSV_HEADER  = ["frame_idx", "timestamp", "image_name", "steering_angle", "command", "speed"]


# ============================================================
# 2. CAPTURA DE CAMARA  (reutilizada de la Act. 2.1 / 3.1 / 4.2)
# ============================================================

def get_image(camera):
    """Extrae la imagen de la camara como matriz Numpy BGRA (alto x ancho x 4)."""
    raw = camera.getImage()
    if raw is None:
        return None
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )


# ============================================================
# 3. PERSISTENCIA DEL DATASET
# ============================================================

def init_dataset():
    """
    Crea las carpetas del dataset y el CSV con encabezado si no existen.
    Si el CSV ya existe, se respeta (modo append) para acumular varias sesiones
    de conduccion de los distintos integrantes del equipo.
    """
    os.makedirs(IMG_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)
        print(f"[DATASET] CSV creado en {CSV_PATH}")
    else:
        print(f"[DATASET] CSV existente, se agregan filas en {CSV_PATH}")


def count_existing_rows():
    """Cuenta cuantas filas de datos hay ya en el CSV (para continuar la numeracion)."""
    if not os.path.exists(CSV_PATH):
        return 0
    with open(CSV_PATH, "r") as f:
        return max(0, sum(1 for _ in f) - 1)   # -1 por el encabezado


def save_sample(camera, frame_idx, steering, command, speed):
    """
    Guarda la imagen actual de la camara y registra la fila correspondiente en el CSV.

    El nombre de archivo embebe el comando para inspeccion visual rapida del balance
    del dataset: <COMANDO>_<frame>.png. Se usa camera.saveImage (mismo metodo que en
    las actividades previas) para que el PNG quede en el espacio de color correcto.
    """
    image_name = f"{CMD_NAMES[command]}_{frame_idx:07d}.png"
    image_path = os.path.join(IMG_DIR, image_name)
    # quality=100 -> sin perdida apreciable; el segundo argumento es la calidad PNG/JPG.
    camera.saveImage(image_path, 100)

    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            frame_idx,
            f"{time.time():.3f}",
            image_name,
            f"{steering:.5f}",
            command,                       # indice entero 0..3 (clave: CMD_*)
            f"{speed:.2f}",
        ])


# ============================================================
# 4. INTERFAZ EN PANTALLA (Display de Webots)
# ============================================================

def draw_overlay(display, bgra, command, steering, speed, recording, saved):
    """
    Dibuja la imagen de la camara en el Display integrado y superpone el estado:
    comando activo, angulo, velocidad, si esta grabando y cuantas muestras lleva.
    Sirve de retroalimentacion al conductor para mantener el dataset balanceado.
    """
    if display is None or bgra is None:
        return
    # Webots Display espera ARGB empaquetado; convertimos desde BGRA.
    h, w, _ = bgra.shape
    argb = (bgra[:, :, 3].astype(np.uint32) << 24 |
            bgra[:, :, 2].astype(np.uint32) << 16 |
            bgra[:, :, 1].astype(np.uint32) << 8  |
            bgra[:, :, 0].astype(np.uint32))
    ref = display.imageNew(argb.tobytes(), Display.ARGB, w, h)
    display.imagePaste(ref, 0, 0, False)
    display.imageDelete(ref)

    estado = "REC" if recording else "PAUSA"
    display.setColor(0xFF0000 if recording else 0xAAAAAA)
    display.drawText(f"{estado}  CMD:{CMD_NAMES[command]}", 4, 4)
    display.setColor(0xFFFFFF)
    display.drawText(f"ang:{steering:+.2f}  vel:{speed:4.1f}  n:{saved}", 4, 16)


# ============================================================
# 5. MAIN — BUCLE PRINCIPAL DEL CONTROLADOR
# ============================================================

def main():
    # --- Inicializacion de Webots (patron canonico de las actividades previas) ---
    driver = Driver()
    timestep = int(driver.getBasicTimeStep())

    camera = driver.getDevice("camera")
    camera.enable(timestep)

    keyboard = Keyboard()
    keyboard.enable(timestep)

    # El Display es opcional: si el mundo no lo tiene, el controlador sigue operando.
    try:
        display = driver.getDevice("display_image")
    except Exception:
        display = None

    init_dataset()

    # --- Estado del controlador ---
    steering = 0.0                  # angulo actual del volante (rad)
    speed = COLLECT_SPEED           # velocidad crucero (km/h)
    command = CMD_FOLLOW            # comando de navegacion latente
    recording = True               # se inicia grabando
    frame_idx = count_existing_rows()   # continua la numeracion de sesiones previas
    saved = frame_idx
    step_counter = 0
    last_key_time = 0.0             # para el anti-rebote de teclas de un disparo

    print("=" * 70)
    print(" RECOLECCION CIL — Mundo #1")
    print("  Flechas: direccion/velocidad | R: enderezar | 1-4: comando")
    print("  G: grabar on/off | S: detener")
    print("=" * 70)

    while driver.step() != -1:
        now = time.time()
        key = keyboard.getKey()

        # --- Teclas continuas (se pueden mantener presionadas) ---
        if key == Keyboard.LEFT:
            steering = max(-MAX_ANGLE, steering - ANGLE_INCR)
        elif key == Keyboard.RIGHT:
            steering = min(MAX_ANGLE, steering + ANGLE_INCR)
        elif key == Keyboard.UP:
            speed = min(MAX_SPEED, speed + SPEED_INCR)
        elif key == Keyboard.DOWN:
            speed = max(0.0, speed - SPEED_INCR)

        # --- Teclas de un solo disparo (con anti-rebote) ---
        if key != -1 and (now - last_key_time) > DEBOUNCE_TIME:
            if key == ord('R') or key == ord('r'):
                steering = 0.0
            elif key == ord('G') or key == ord('g'):
                recording = not recording
                print(f"[REC] {'ON' if recording else 'OFF'}")
            elif key == ord('S') or key == ord('s'):
                speed = 0.0
            elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
                command = key - ord('1')   # '1'->0 ... '4'->3
                print(f"[CMD] {CMD_NAMES[command]}")
            last_key_time = now

        # --- Aplicar comandos al vehiculo ---
        driver.setSteeringAngle(steering)
        driver.setCruisingSpeed(speed)

        # --- Captura automatica del dataset ---
        # Se guarda solo si estamos grabando, el auto se mueve (evita miles de
        # frames identicos detenido) y toca el periodo de muestreo.
        bgra = get_image(camera)
        step_counter += 1
        if recording and speed > 0.1 and (step_counter % CAPTURE_EVERY == 0):
            save_sample(camera, frame_idx, steering, command, speed)
            frame_idx += 1
            saved = frame_idx

        draw_overlay(display, bgra, command, steering, speed, recording, saved)

    print(f"[DATASET] Sesion finalizada. Total de muestras: {saved}")


if __name__ == "__main__":
    main()
