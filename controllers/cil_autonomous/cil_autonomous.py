r"""
===============================================================================
  Proyecto Final — Conditional Imitation Learning (CIL)  ·  MR4010.10
  Controlador AUTONOMO de evaluacion (Mundo #2)
===============================================================================

  Proposito
  ---------
  Conducir el vehiculo de forma autonoma en el Mundo #2 usando el modelo CIL
  entrenado en el Mundo #1. El operador entrega comandos de navegacion por teclado
  (FOLLOW / LEFT / STRAIGHT / RIGHT) para cubrir distintas rutas; el modelo predice
  el angulo de direccion condicionado a ese comando. La velocidad es constante (baja,
  carril derecho), salvo cuando una capa de seguridad la modifica.

  Arbitraje por prioridad (de mayor a menor):
      1. PEATON + freno de emergencia : deteccion por nodo Recognition de la camara
         en coordinacion con el LiDAR (y confirmacion opcional con HOG+SVM). Freno total.
      2. DISTANCIA con RADAR          : mantiene una distancia de umbral al vehiculo mas
         proximo; reduce la velocidad de forma proporcional y se detiene por debajo del umbral.
      3. EVASION de obstaculo         : ante un objeto estatico en el carril, ejecuta un
         seguimiento de pared derecha (reutilizado de la Act. 4.2) que sobreescribe la direccion.
      4. CIL (nominal)                : el angulo lo da el modelo segun el comando activo.

  Reglas de la tarea: favorecer el carril derecho, sin vueltas en U; el operador
  reposiciona el vehiculo en el origen de cada ruta antes de iniciar la grabacion.

  Reutilizacion (actividades previas):
      - get_image / process_lidar / deteccion por Recognition .... Act. 4.2 (evasion_obstaculos)
      - detect_pedestrian (HOG + SVM) ........................... Act. 3.1 (navegacion_autonoma_svm)
      - freno de emergencia (setBrakeIntensity) ................. Act. 3.1
      - inferencia TFLite (clase + fallback de runtime) ......... Act. 4.x (navegacion_autonoma_cnn)
  Componente NUEVO de este proyecto: lectura del nodo Radar para la distancia de umbral.

  Equipo:
      Antonio Olvera Donlucas          A01795617
      Carlos Monir Radovich Saad       A01797569
      Andres Roberto Osuna Gonzalez    A01796264
      Oscar Alberto Ramirez Anaya      A01795438
===============================================================================
"""

import os
import math
import numpy as np
import cv2

# --- Runtime de inferencia TFLite (con respaldos; patron de la Act. 4.x) ---
CIL_AVAILABLE = True
_Interpreter = None
try:
    from ai_edge_litert.interpreter import Interpreter as _Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter as _Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter as _Interpreter
        except ImportError:
            print("[WARN] Sin runtime TFLite (ai-edge-litert / tflite-runtime / tensorflow). "
                  "La inferencia CIL queda desactivada; el auto solo aplicara seguridad.")
            CIL_AVAILABLE = False

# --- HOG + SVM para confirmar peatones (opcional) ---
SVM_AVAILABLE = True
try:
    import joblib
    from skimage.feature import hog
except ImportError:
    SVM_AVAILABLE = False

# --- Imports de Webots ---
from controller import Display, Keyboard
from vehicle import Driver


# ============================================================
# 1. CONSTANTES
# ============================================================

# --- Modelo / preprocesamiento (DEBE coincidir con train_cil.py) ---
IMG_H, IMG_W = 88, 200   # debe coincidir con train_cil.py (entrada estilo Codevilla)
NUM_COMMANDS = 4
CMD_FOLLOW, CMD_LEFT, CMD_STRAIGHT, CMD_RIGHT = 0, 1, 2, 3
CMD_NAMES = {CMD_FOLLOW: "FOLLOW", CMD_LEFT: "LEFT",
             CMD_STRAIGHT: "STRAIGHT", CMD_RIGHT: "RIGHT"}

# Teclas para dar el comando durante la conduccion autonoma (Q/W/E = izq/recto/der,
# F = seguir; tambien 1-4 como alias).
CMD_KEYS = {
    ord('Q'): CMD_LEFT, Keyboard.UP: CMD_STRAIGHT, ord('W'): CMD_STRAIGHT,
    ord('E'): CMD_RIGHT, ord('F'): CMD_FOLLOW,
    ord('1'): CMD_FOLLOW, ord('2'): CMD_LEFT, ord('3'): CMD_STRAIGHT, ord('4'): CMD_RIGHT,
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "cil_model.tflite")
SVM_MODEL_PATH = os.path.join(SCRIPT_DIR, "svm_pedestrian_model.joblib")

# --- Velocidad ---
CRUISE_SPEED = 25.0     # km/h — crucero en el carril derecho de baja velocidad
SLOW_SPEED = 10.0       # km/h — velocidad reducida al aproximarse a un riesgo
MAX_ANGLE = 0.5         # rad — limite del volante (igual que en entrenamiento)

# --- LiDAR (Sick LMS 291) ---
LIDAR_HALF_AREA = 20    # indices a cada lado del centro que se analizan
LIDAR_MAX_DIST = 20.0   # m — el LiDAR ignora cualquier cosa mas alla

# --- Peaton (freno de emergencia) ---
PED_BRAKE_DIST = 8.0    # m — bajo esta distancia frontal con peaton -> freno total

# --- Radar (distancia de umbral al vehiculo mas proximo)  [COMPONENTE NUEVO] ---
# DIST_UMBRAL es el valor que el video de evidencia debe declarar explicitamente.
RADAR_DIST_UMBRAL = 12.0   # m — por debajo de esto el auto se detiene
RADAR_SLOW_BAND = 8.0      # m — banda extra sobre el umbral donde reduce velocidad de forma proporcional
RADAR_AZIMUTH_MAX = 0.20   # rad — solo objetivos casi al frente (descarta carriles vecinos)

# --- Evasion de obstaculo estatico (reutilizado de la Act. 4.2) ---
EVADE_APPROACH_DIST = 14.0   # m — objeto estatico en el carril mas cerca de esto -> evadir
EVADE_SPEED = 10.0           # km/h durante la maniobra
EVADE_STEER = -0.30          # rad — giro a la izquierda para salir del carril
WALL_CLEAR_DIST = 4.0        # m — sensor lateral derecho >= esto -> costado libre
RAIL_SAFE = 3.0              # m — ds_left < esto -> empuje anti-barandal hacia la derecha
RAIL_GAIN = 0.30             # rad/m — ganancia del empuje anti-barandal
MIN_EVADE_STEPS = 40         # pasos minimos dentro de la evasion antes de reincorporarse
MAX_EVADE_STEPS = 800        # salida de seguridad de la evasion

DEBOUNCE_STEPS = 4           # pasos de anti-rebote para teclas de comando
DEBUG_EVERY = 30


# ============================================================
# 2. INFERENCIA CIL (TFLite, dos entradas: imagen + comando one-hot)
# ============================================================

class CILDriver:
    """
    Envoltura del interprete TFLite del modelo CIL ramificado. Recibe una imagen
    BGRA de la camara y un indice de comando, y devuelve el angulo de direccion.
    El preprocesamiento es identico al de train_cil.preprocess para que la
    inferencia coincida con el entrenamiento.
    """

    def __init__(self, model_path):
        self.interpreter = _Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        self.out = self.interpreter.get_output_details()[0]
        # El modelo tiene dos entradas; se distinguen por su forma (imagen=4D, comando=2D).
        self.img_in = next(d for d in inputs if len(d["shape"]) == 4)
        self.cmd_in = next(d for d in inputs if len(d["shape"]) == 2)

    @staticmethod
    def preprocess(bgra):
        """BGRA -> recorte de carretera -> RGB -> resize (66x200) -> normalizado [0,1]."""
        bgr = bgra[:, :, :3]
        h = bgr.shape[0]
        crop = bgr[int(0.40 * h):int(0.90 * h), :, :]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        return resized.astype(np.float32) / 255.0

    def predict(self, bgra, command):
        x = np.expand_dims(self.preprocess(bgra), axis=0)             # (1,66,200,3)
        onehot = np.zeros((1, NUM_COMMANDS), dtype=np.float32)
        onehot[0, command] = 1.0
        self.interpreter.set_tensor(self.img_in["index"], x)
        self.interpreter.set_tensor(self.cmd_in["index"], onehot)
        self.interpreter.invoke()
        steering = float(self.interpreter.get_tensor(self.out["index"]).ravel()[0])
        return float(np.clip(steering, -MAX_ANGLE, MAX_ANGLE))


# ============================================================
# 3. SENSORES Y PERCEPCION (reutilizados de actividades previas)
# ============================================================

def get_image(camera):
    """Imagen de la camara como matriz Numpy BGRA (reutilizado de la Act. 2.1/3.1/4.2)."""
    raw = camera.getImage()
    if raw is None:
        return None
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )


def process_lidar(lidar):
    """
    Procesa el LiDAR Sick LMS 291 en la zona central (+/-LIDAR_HALF_AREA indices).
    Retorna (angulo, distancia) del obstaculo frontal mas cercano, o (None, None).
    Reutilizado sin cambios de la Act. 4.2.
    """
    range_data = lidar.getRangeImage()
    if not range_data:
        return None, None
    n = len(range_data)
    center = n // 2
    sumx, count, dist_sum = 0, 0, 0.0
    for x in range(center - LIDAR_HALF_AREA, center + LIDAR_HALF_AREA):
        r = range_data[x]
        if r <= LIDAR_MAX_DIST and not math.isinf(r) and not math.isnan(r):
            sumx += x
            count += 1
            dist_sum += r
    if count == 0:
        return None, None
    avg_angle = (sumx / count / n - 0.5) * lidar.getFov()
    return avg_angle, dist_sum / count


def detect_pedestrian_recognition(camera):
    """
    Usa el nodo Recognition de la camara para detectar un peaton (model 'pedestrian')
    en el tercio central de la imagen. Adaptado de detect_bus_ahead (Act. 4.2),
    cambiando el filtro de 'autobus' a 'pedestrian'.
    Retorna True si hay un peaton bloqueando el carril.
    """
    objects = camera.getRecognitionObjects()
    cam_cx = camera.getWidth() / 2.0
    for obj in objects:
        modelo = obj.getModel()
        # En algunos PROTO el peaton se identifica por 'pedestrian'; se aceptan variantes.
        if modelo in ("pedestrian", "Pedestrian", "human", "Pedestrian.proto"):
            pos = obj.getPositionOnImage()
            if abs(pos[0] - cam_cx) < cam_cx * 0.8:
                return True
    return False


def confirm_pedestrian_svm(bgra, svm_model, lidar_dist):
    """
    Confirmacion opcional con HOG + SVM (Act. 3.1) usando ventana adaptada a la
    distancia del LiDAR. Solo se invoca si hay modelo cargado. Devuelve True si el
    SVM tambien clasifica el objeto como peaton.
    """
    if svm_model is None or not SVM_AVAILABLE:
        return True   # sin SVM, se confia en el Recognition
    h, w = bgra.shape[:2]
    roi = bgra[int(h * 0.25):int(h * 0.85), :, :3]
    d = lidar_dist if lidar_dist is not None else 10.0
    win, step = (32, 20) if d > 14 else (48, 24) if d > 8 else (64, 32)
    rh, rw = roi.shape[:2]
    for y in range(0, rh - win + 1, step):
        for x in range(0, rw - win + 1, step):
            window = cv2.resize(roi[y:y + win, x:x + win], (64, 64))
            gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
            feats = hog(gray, orientations=11, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), visualize=False, feature_vector=True)
            if svm_model.decision_function([feats])[0] > 0.55:
                return True
    return False


def read_radar_nearest(radar):
    """
    [COMPONENTE NUEVO] Lee el nodo Radar y devuelve la distancia (m) al objetivo
    frontal mas proximo dentro de +/-RADAR_AZIMUTH_MAX rad, o None si no hay objetivos.
    El Radar de Webots entrega distancia, velocidad relativa y azimut por objetivo,
    lo que lo hace ideal para mantener la distancia al vehiculo de adelante.
    """
    if radar is None:
        return None
    targets = radar.getTargets()
    if not targets:
        return None
    mejor = None
    for t in targets:
        # getAzimuth() / getDistance() son metodos de RadarTarget en Webots.
        if abs(t.getAzimuth()) <= RADAR_AZIMUTH_MAX:
            if mejor is None or t.getDistance() < mejor:
                mejor = t.getDistance()
    return mejor


# ============================================================
# 4. PANTALLA DE A BORDO
# ============================================================

def mostrar_en_display(display, bgra, texto):
    """Envia la imagen de la camara al Display y superpone una linea de estado."""
    if display is None or bgra is None:
        return
    rgb = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB)
    ref = display.imageNew(rgb.tobytes(), Display.RGB,
                           width=rgb.shape[1], height=rgb.shape[0])
    display.imagePaste(ref, 0, 0, False)
    display.imageDelete(ref)
    display.setColor(0xFFFFFF)
    display.drawText(texto, 4, 4)


# ============================================================
# 5. CARGA DEFENSIVA DE DISPOSITIVOS
# ============================================================

def get_device_opt(driver, name, enable_ts=None, point_cloud=False):
    """
    Obtiene un dispositivo por nombre sin abortar si no existe en el mundo.
    Permite que el controlador degrade con elegancia (p.ej. si el mundo no tiene
    radar o sensores laterales, esas capas de seguridad simplemente no se activan).
    """
    try:
        dev = driver.getDevice(name)
        if dev is None:
            return None
        if enable_ts is not None and hasattr(dev, "enable"):
            dev.enable(enable_ts)
        if point_cloud and hasattr(dev, "enablePointCloud"):
            dev.enablePointCloud()
        return dev
    except Exception:
        return None


# ============================================================
# 6. MAIN — BUCLE PRINCIPAL DEL CONTROLADOR
# ============================================================

def main():
    # --- Modelo CIL ---
    cil = None
    if CIL_AVAILABLE and os.path.exists(MODEL_PATH):
        cil = CILDriver(MODEL_PATH)
        print(f"[INFO] Modelo CIL cargado: {MODEL_PATH}")
    else:
        print(f"[WARN] Modelo CIL no disponible en {MODEL_PATH}. "
              "El auto avanzara recto bajo las capas de seguridad.")

    # --- Modelo SVM de peatones (opcional) ---
    svm_model = None
    if SVM_AVAILABLE and os.path.exists(SVM_MODEL_PATH):
        # Artefacto local de confianza: el .joblib es el SVM entrenado por el propio
        # equipo en la Act. 3.1 (no proviene de una fuente externa no confiable).
        svm_model = joblib.load(SVM_MODEL_PATH)
        print(f"[INFO] Modelo SVM de peatones cargado: {SVM_MODEL_PATH}")

    # --- Inicializacion de Webots ---
    driver = Driver()
    timestep = int(driver.getBasicTimeStep())

    camera = driver.getDevice("camera")
    camera.enable(timestep)
    try:
        camera.recognitionEnable(timestep)   # nodo Recognition para peatones
    except Exception:
        print("[WARN] La camara no tiene nodo Recognition; se usara solo SVM/LiDAR para peatones.")

    lidar = get_device_opt(driver, "Sick LMS 291", enable_ts=timestep, point_cloud=True)
    radar = get_device_opt(driver, "radar", enable_ts=timestep)
    keyboard = Keyboard(); keyboard.enable(timestep)
    display = get_device_opt(driver, "display_image")

    # Sensores laterales de la Act. 4.2 (si el mundo los incluye, habilitan la evasion).
    ds_left = get_device_opt(driver, "ds_left", enable_ts=timestep)
    ds_right_front = get_device_opt(driver, "ds_right_front", enable_ts=timestep)
    ds_right_rear = get_device_opt(driver, "ds_right_rear", enable_ts=timestep)

    if radar is None:
        print("[WARN] Sin nodo 'radar' en el mundo: agregue Radar al sensorsSlotFront "
              "del BmwX5 (junto al LiDAR) para habilitar el mantenimiento de distancia de umbral.")

    # --- Estado del controlador ---
    command = CMD_FOLLOW
    last_cmd_step = -DEBOUNCE_STEPS
    evade_active = False
    evade_steps = 0
    step = 0

    print("=" * 70)
    print(" CONTROLADOR AUTONOMO CIL — Mundo #2")
    print("  Comandos por teclado: 1=FOLLOW 2=LEFT 3=STRAIGHT 4=RIGHT")
    print(f"  Distancia de umbral del radar: {RADAR_DIST_UMBRAL:.1f} m")
    print("=" * 70)

    while driver.step() != -1:
        step += 1
        bgra = get_image(camera)

        # --- Comando de navegacion por teclado (latched, con anti-rebote) ---
        # Se vacia la cola de teclas (getKey devuelve una por llamada) para no perder
        # el comando si se pulsa junto con otra tecla.
        k = keyboard.getKey()
        while k != -1:
            kk = k & 0xFFFF
            if kk in CMD_KEYS and (step - last_cmd_step) > DEBOUNCE_STEPS:
                command = CMD_KEYS[kk]
                last_cmd_step = step
                print(f"[CMD] {CMD_NAMES[command]}")
            k = keyboard.getKey()

        # --- Percepcion ---
        lidar_angle, lidar_dist = process_lidar(lidar) if lidar else (None, None)
        radar_dist = read_radar_nearest(radar)

        # --- Direccion nominal del modelo CIL ---
        if cil is not None and bgra is not None:
            steering = cil.predict(bgra, command)
        else:
            steering = 0.0
        speed = CRUISE_SPEED
        motivo = f"CIL:{CMD_NAMES[command]}"

        # ===== ARBITRAJE DE SEGURIDAD (prioridad descendente) =====

        # (1) PEATON + freno de emergencia
        peaton = False
        if bgra is not None:
            peaton = detect_pedestrian_recognition(camera)
            if peaton and svm_model is not None:
                peaton = confirm_pedestrian_svm(bgra, svm_model, lidar_dist)
        if peaton:
            motivo = "PEATON"
            if lidar_dist is not None and lidar_dist < PED_BRAKE_DIST:
                speed = 0.0
                driver.setSteeringAngle(steering)
                driver.setCruisingSpeed(0.0)
                _set_brake(driver, 1.0)
                _debug(step, motivo, steering, 0.0, lidar_dist, radar_dist)
                mostrar_en_display(display, bgra, "PEATON - FRENO")
                continue
            else:
                speed = SLOW_SPEED   # peaton lejos: precaucion

        # (2) DISTANCIA con RADAR (vehiculo mas proximo)
        elif radar_dist is not None and radar_dist < (RADAR_DIST_UMBRAL + RADAR_SLOW_BAND):
            if radar_dist < RADAR_DIST_UMBRAL:
                # Por debajo del umbral: detenerse.
                motivo = f"RADAR<{RADAR_DIST_UMBRAL:.0f}m"
                driver.setSteeringAngle(steering)
                driver.setCruisingSpeed(0.0)
                _set_brake(driver, 1.0)
                _debug(step, motivo, steering, 0.0, lidar_dist, radar_dist)
                mostrar_en_display(display, bgra, f"RADAR {radar_dist:.1f}m - ALTO")
                continue
            else:
                # En la banda de aproximacion: reducir velocidad proporcionalmente.
                motivo = "RADAR aprox"
                frac = (radar_dist - RADAR_DIST_UMBRAL) / RADAR_SLOW_BAND   # 0..1
                speed = SLOW_SPEED + frac * (CRUISE_SPEED - SLOW_SPEED)

        # (3) EVASION de obstaculo estatico (no peaton, no vehiculo gestionado por radar)
        if not peaton:
            obstaculo_frente = (lidar_dist is not None and lidar_dist < EVADE_APPROACH_DIST
                                and radar_dist is None and ds_right_front is not None)
            if obstaculo_frente and not evade_active:
                evade_active = True
                evade_steps = 0
                print(f"[EVADE] Obstaculo a {lidar_dist:.1f} m -> inicia evasion")
            if evade_active:
                steering, speed, evade_active = _wall_following(
                    ds_left, ds_right_front, ds_right_rear, evade_steps)
                evade_steps += 1
                motivo = "EVADE"
                if evade_steps > MAX_EVADE_STEPS:
                    evade_active = False

        # --- Empuje anti-barandal (en todos los estados, si hay sensor izquierdo) ---
        if ds_left is not None:
            dl = ds_left.getValue()
            if dl < RAIL_SAFE:
                steering += RAIL_GAIN * (RAIL_SAFE - dl)
        steering = float(np.clip(steering, -MAX_ANGLE, MAX_ANGLE))

        # --- Aplicar comandos nominales ---
        driver.setSteeringAngle(steering)
        driver.setCruisingSpeed(speed)
        _set_brake(driver, 0.0)

        if step % DEBUG_EVERY == 0:
            _debug(step, motivo, steering, speed, lidar_dist, radar_dist)
        mostrar_en_display(display, bgra,
                           f"{motivo} a:{steering:+.2f} v:{speed:4.1f}")


# ============================================================
# 7. UTILIDADES
# ============================================================

def _set_brake(driver, intensity):
    """Aplica el freno de servicio con proteccion (algunos PROTO no exponen el metodo)."""
    try:
        driver.setBrakeIntensity(intensity)
    except Exception:
        pass


def _wall_following(ds_left, ds_right_front, ds_right_rear, steps):
    """
    Maniobra compacta de seguimiento de pared derecha para evadir un obstaculo
    estatico (reutilizada de la Act. 4.2). Sale del carril girando a la izquierda y
    se reincorpora cuando el costado derecho queda libre tras un minimo de pasos.
    Retorna (steering, speed, evade_active).
    """
    rf = ds_right_front.getValue() if ds_right_front is not None else 99.0
    rr = ds_right_rear.getValue() if ds_right_rear is not None else 99.0

    # Reincorporacion: costado derecho libre tras el minimo de pasos.
    if steps > MIN_EVADE_STEPS and rf >= WALL_CLEAR_DIST and rr >= WALL_CLEAR_DIST:
        return 0.0, EVADE_SPEED, False

    # Fase de salida/paso: girar a la izquierda para rodear el obstaculo.
    steer = EVADE_STEER
    # Si ya rebasamos el frente del obstaculo, enderezar gradualmente para volver.
    if rf >= WALL_CLEAR_DIST:
        steer = EVADE_STEER * 0.4
    return steer, EVADE_SPEED, True


def _debug(step, motivo, steering, speed, lidar_dist, radar_dist):
    ld = f"{lidar_dist:.1f}" if lidar_dist is not None else "--"
    rd = f"{radar_dist:.1f}" if radar_dist is not None else "--"
    print(f"[{step:5d}] {motivo:12s} ang={steering:+.3f} vel={speed:4.1f} "
          f"lidar={ld} radar={rd}")


if __name__ == "__main__":
    main()
