"""
===============================================================================
  Proyecto Final — Conditional Imitation Learning (CIL)  ·  MR4010.10
  Controlador de RECOLECCION ASISTIDA de datos (Mundo #1)
===============================================================================

  Idea
  ----
  Recolectar el dataset de CIL con el minimo esfuerzo humano. En lugar de conducir
  todo manualmente (dificil y desbalanceado), el vehiculo **se conduce solo siguiendo
  el carril** con el pipeline de vision + PID reutilizado de la Actividad 2.1 / 4.2.
  El operador unicamente **da el comando de navegacion** en cada interseccion
  (izquierda / recto / derecha); el controlador ejecuta el giro y **etiqueta cada
  imagen automaticamente** con (angulo aplicado, comando).

  Asi:
    - El manejo en linea recta / curvas (comando FOLLOW) es automatico y suave.
    - Los giros se asisten con un sesgo de direccion + posibilidad de corregir con
      las flechas.
    - El angulo guardado es el que realmente aplica el vehiculo (experto), mas
      consistente que la conduccion manual temblorosa.

  Pipeline de vision (reutilizado sin cambios de la Act. 2.1 / 4.2):
      Camara -> Grises -> Canny(50,150) -> ROI trapezoidal -> Hough -> error -> EMA -> PID

  Controles de teclado
  --------------------
    Comando (UN toque por interseccion; FOLLOW es automatico y regresa solo):
        Q             = LEFT     (girar a la izquierda)
        Flecha ARRIBA = STRAIGHT (cruzar derecho la interseccion)
        E             = RIGHT    (girar a la derecha)
    Flecha IZQUIERDA / DERECHA : corrigen la direccion manualmente (se suma al PID).
    Flecha ABAJO               : frena mientras se mantiene.
    G                          : activa / desactiva la grabacion.

  Salida (dataset)
  ----------------
    dataset/IMG/<comando>_<frame>.png   imagenes capturadas
    dataset/driving_log.csv             frame_idx, timestamp, image_name,
                                        steering_angle, command, speed

  La velocidad se mantiene constante y baja (<=30 km/h) y NO se entrena.

  Equipo:
      Antonio Olvera Donlucas          A01795617
      Carlos Monir Radovich Saad       A01797569
      Andres Roberto Osuna Gonzalez    A01796264
      Oscar Alberto Ramirez Anaya      A01795438
===============================================================================
"""

import os
import csv
import time
import numpy as np
import cv2

from controller import Display, Keyboard
from vehicle import Driver


# ============================================================
# 1. CONSTANTES
# ============================================================

# --- Direccion ---
MAX_ANGLE = 0.5        # rad — angulo maximo del volante
MANUAL_INCR = 0.06     # rad — correccion manual por pulsacion de flecha izq/der

# --- Velocidad ---
COLLECT_SPEED = 30.0   # km/h — crucero constante (recto)
TURN_SPEED = 16.0      # km/h — mas lento durante un giro (mejor control)
SPEED_INCR = 5.0       # km/h — decremento al frenar con la flecha abajo

# --- PID de seguimiento de carril (reutilizado de la Act. 2.1 / 4.2) ---
KP = 0.006             # proporcional (algo menor que 0.008 por la camara 320x160)
KI = 0.0               # integral (no usada)
KD = 0.012             # derivativo
EMA_ALPHA = 0.6        # suavizado exponencial del error

# --- Ejecucion asistida del giro ---
TURN_BIAS = 0.30       # rad — sesgo de direccion que se suma durante LEFT/RIGHT

# --- Comandos de navegacion ---
CMD_FOLLOW, CMD_LEFT, CMD_STRAIGHT, CMD_RIGHT = 0, 1, 2, 3
CMD_NAMES = {CMD_FOLLOW: "FOLLOW", CMD_LEFT: "LEFT",
             CMD_STRAIGHT: "STRAIGHT", CMD_RIGHT: "RIGHT"}
# Teclas: Q/E a la mano izquierda; flecha ARRIBA = recto. Alias numericos 1-4.
CMD_KEYS = {
    ord('Q'): CMD_LEFT, Keyboard.UP: CMD_STRAIGHT, ord('W'): CMD_STRAIGHT,
    ord('E'): CMD_RIGHT, ord('F'): CMD_FOLLOW,
    ord('1'): CMD_FOLLOW, ord('2'): CMD_LEFT, ord('3'): CMD_STRAIGHT, ord('4'): CMD_RIGHT,
}
COMMAND_HOLD_SECONDS = 4.0   # el comando de giro dura esto y regresa solo a FOLLOW

# --- Captura ---
CAPTURE_EVERY = 3      # guardar 1 imagen cada N pasos
DEBOUNCE_TIME = 0.10   # s — anti-rebote de teclas de un disparo

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
IMG_DIR     = os.path.join(DATASET_DIR, "IMG")
CSV_PATH    = os.path.join(DATASET_DIR, "driving_log.csv")
CSV_HEADER  = ["frame_idx", "timestamp", "image_name", "steering_angle", "command", "speed"]


# ============================================================
# 2. VISION DE CARRIL  (reutilizado de la Act. 2.1 / 4.2)
# ============================================================

def get_image(camera):
    """Imagen de la camara como matriz Numpy BGRA (alto x ancho x 4)."""
    raw = camera.getImage()
    if raw is None:
        return None
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )


def procesar_lineas(bgra):
    """Grises -> Canny -> ROI trapezoidal -> Hough. Retorna las lineas detectadas."""
    gray = cv2.cvtColor(bgra, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    h, w = edges.shape
    mask = np.zeros_like(edges)
    poly = np.array([[
        (0, h), (int(0.2 * w), int(0.5 * h)),
        (int(0.8 * w), int(0.5 * h)), (w, h),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, poly, 255)
    return cv2.HoughLinesP(cv2.bitwise_and(edges, mask), rho=1, theta=np.pi / 180,
                           threshold=15, minLineLength=8, maxLineGap=5)


def calcular_error_direccion(lines, setpoint):
    """Distancia (px) del centro del carril al centro de la camara; None si no hay lineas."""
    if lines is None:
        return None
    cand = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > 3 * abs(y2 - y1):   # descartar horizontales
            continue
        cand.append((x1 + x2) / 2.0 - setpoint)
    return min(cand, key=abs) if cand else None


# ============================================================
# 3. PERSISTENCIA DEL DATASET
# ============================================================

def init_dataset():
    os.makedirs(IMG_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)
        print(f"[DATASET] CSV creado en {CSV_PATH}")
    else:
        print(f"[DATASET] CSV existente, se agregan filas en {CSV_PATH}")


def count_existing_rows():
    if not os.path.exists(CSV_PATH):
        return 0
    with open(CSV_PATH, "r") as f:
        return max(0, sum(1 for _ in f) - 1)


def save_sample(camera, frame_idx, steering, command, speed):
    image_name = f"{CMD_NAMES[command]}_{frame_idx:07d}.png"
    camera.saveImage(os.path.join(IMG_DIR, image_name), 100)
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            frame_idx, f"{time.time():.3f}", image_name,
            f"{steering:.5f}", command, f"{speed:.2f}",
        ])


# ============================================================
# 4. INTERFAZ EN PANTALLA
# ============================================================

def draw_overlay(display, bgra, command, steering, speed, recording, saved):
    if display is None or bgra is None:
        return
    h, w, _ = bgra.shape
    argb = (bgra[:, :, 3].astype(np.uint32) << 24 |
            bgra[:, :, 2].astype(np.uint32) << 16 |
            bgra[:, :, 1].astype(np.uint32) << 8 |
            bgra[:, :, 0].astype(np.uint32))
    ref = display.imageNew(argb.tobytes(), Display.ARGB, w, h)
    display.imagePaste(ref, 0, 0, False)
    display.imageDelete(ref)
    display.setColor(0xFF0000 if recording else 0xAAAAAA)
    display.drawText(f"{'REC' if recording else 'PAUSA'}  CMD:{CMD_NAMES[command]}", 4, 4)
    display.setColor(0xFFFFFF)
    display.drawText(f"ang:{steering:+.2f}  vel:{speed:4.1f}  n:{saved}", 4, 16)


# ============================================================
# 5. MAIN
# ============================================================

def main():
    driver = Driver()
    timestep = int(driver.getBasicTimeStep())
    camera = driver.getDevice("camera")
    camera.enable(timestep)
    keyboard = Keyboard()
    keyboard.enable(timestep)
    try:
        display = driver.getDevice("display_image")
    except Exception:
        display = None

    init_dataset()
    setpoint = camera.getWidth() / 2.0

    # --- Estado ---
    command = CMD_FOLLOW
    recording = True
    frame_idx = count_existing_rows()
    saved = frame_idx
    step_counter = 0
    last_key_time = 0.0
    command_set_time = 0.0
    smoothed_error = 0.0     # error EMA del PID
    prev_error = 0.0
    manual_speed = COLLECT_SPEED

    print("=" * 70)
    print(" RECOLECCION ASISTIDA CIL — Mundo #1  (el auto sigue el carril solo)")
    print("  Comando (un toque):  Q=LEFT   flecha ARRIBA=STRAIGHT   E=RIGHT")
    print("  Correccion manual: <- ->   |   Frenar: flecha ABAJO   |   G: grabar")
    print("  FOLLOW es automatico.")
    print("=" * 70)

    while driver.step() != -1:
        now = time.time()
        bgra = get_image(camera)

        # --- Leer todas las teclas del paso ---
        keys = []
        k = keyboard.getKey()
        while k != -1:
            keys.append(k & 0xFFFF)
            k = keyboard.getKey()

        # --- Comandos (un disparo, con anti-rebote) ---
        if keys and (now - last_key_time) > DEBOUNCE_TIME:
            for kk in keys:
                if kk in (ord('G'), ord('g')):
                    recording = not recording
                    print(f"[REC] {'ON' if recording else 'OFF'}")
                elif kk in CMD_KEYS:
                    command = CMD_KEYS[kk]
                    command_set_time = now
                    print(f"[CMD] {CMD_NAMES[command]}")
            last_key_time = now

        # Auto-retorno a FOLLOW tras cruzar la interseccion
        if command != CMD_FOLLOW and (now - command_set_time) > COMMAND_HOLD_SECONDS:
            command = CMD_FOLLOW
            print("[CMD] FOLLOW (auto)")

        # --- Seguimiento de carril (PID) ---
        lines = procesar_lineas(bgra) if bgra is not None else None
        error = calcular_error_direccion(lines, setpoint)
        if error is not None:
            smoothed_error = EMA_ALPHA * smoothed_error + (1 - EMA_ALPHA) * error
            derivative = smoothed_error - prev_error
            prev_error = smoothed_error
            lane_steer = KP * smoothed_error + KD * derivative
        else:
            # Sin lineas (interseccion abierta): mantener recto
            lane_steer = 0.0
            prev_error = 0.0

        # --- Combinar con el comando de navegacion ---
        if command == CMD_LEFT:
            steering = lane_steer - TURN_BIAS       # sesgo hacia la izquierda
        elif command == CMD_RIGHT:
            steering = lane_steer + TURN_BIAS       # sesgo hacia la derecha
        elif command == CMD_STRAIGHT:
            steering = 0.0                          # cruzar recto
        else:  # FOLLOW
            steering = lane_steer

        # --- Correccion manual opcional (se suma al automatico) ---
        if Keyboard.LEFT in keys:
            steering -= MANUAL_INCR
        if Keyboard.RIGHT in keys:
            steering += MANUAL_INCR
        steering = float(np.clip(steering, -MAX_ANGLE, MAX_ANGLE))

        # --- Velocidad: mas lenta en giros; flecha abajo frena ---
        target = TURN_SPEED if command in (CMD_LEFT, CMD_RIGHT) else COLLECT_SPEED
        if Keyboard.DOWN in keys:
            manual_speed = max(0.0, manual_speed - SPEED_INCR)
        else:
            manual_speed = target
        speed = manual_speed

        driver.setSteeringAngle(steering)
        driver.setCruisingSpeed(speed)

        # --- Captura + etiquetado automatico ---
        step_counter += 1
        if recording and speed > 0.1 and (step_counter % CAPTURE_EVERY == 0):
            save_sample(camera, frame_idx, steering, command, speed)
            frame_idx += 1
            saved = frame_idx

        draw_overlay(display, bgra, command, steering, speed, recording, saved)

    print(f"[DATASET] Sesion finalizada. Total de muestras: {saved}")


if __name__ == "__main__":
    main()
