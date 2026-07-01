r"""
===============================================================================
  Proyecto Final — Conditional Imitation Learning (CIL)  ·  MR4010.10
  Entrenamiento del modelo CIL ramificado (Codevilla et al., arXiv:1710.02410)
  Backbone convolucional inspirado en Bojarski et al. (arXiv:1604.07316).
===============================================================================

  Que hace este script
  --------------------
  1. Carga el dataset de conduccion manual (imagenes + driving_log.csv) generado
     por el controlador `cil_data_collector` en el Mundo #1.
  2. Aplica data augmentation especifico de Behavioral Cloning:
       - flip horizontal con NEGACION del angulo e INTERCAMBIO del comando
         LEFT <-> RIGHT (FOLLOW y STRAIGHT no cambian),
       - jitter de brillo.
     El objetivo es superar las 10 mil imagenes y balancear ambos sentidos.
  3. Entrena un modelo CIL RAMIFICADO: un backbone CNN comun extrae caracteristicas
     de la imagen y cuatro ramas densas (una por comando de navegacion) predicen el
     angulo de direccion. El comando activo selecciona, mediante una mascara one-hot,
     que rama produce la salida. La velocidad NO entra al modelo (asi lo pide el
     enunciado: la velocidad se mantiene constante y no se entrena).
  4. Exporta el modelo a `.keras` y `.tflite` (lo que carga el controlador del
     Mundo #2) y verifica la paridad Keras vs TFLite.

  Estructura del modelo
  ---------------------
      imagen (66,200,3) ---> [ Conv32 x2 -> Pool -> Dropout ]
                              [ Conv64 x2 -> Pool -> Dropout ]
                              [ Conv64 x2 -> Pool -> Dropout ]
                              Flatten -> Dense(256) -> z
      z ---> Rama FOLLOW   : Dense(128) -> Dense(1)  \
      z ---> Rama LEFT     : Dense(128) -> Dense(1)   \  apila -> (N,4)
      z ---> Rama STRAIGHT : Dense(128) -> Dense(1)   /  * mascara one-hot del comando
      z ---> Rama RIGHT    : Dense(128) -> Dense(1)  /   -> suma -> steering (N,1)

  Uso
  ---
      python train_cil.py --data_dir ./cil_dataset --epochs 40
  (En Google Colab se ejecuta a traves de cil_colab.ipynb, que primero clona el
   dataset desde GitHub con `!git clone ...`).

  Equipo:
      Antonio Olvera Donlucas          A01795617
      Carlos Monir Radovich Saad       A01797569
      Andres Roberto Osuna Gonzalez    A01796264
      Oscar Alberto Ramirez Anaya      A01795438
===============================================================================
"""

import os
import csv
import argparse
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# =============================================================================
# CONSTANTES
# =============================================================================

IMG_H, IMG_W = 88, 200      # entrada estilo Codevilla (alto x ancho) tras recortar cielo/cofre
NUM_COMMANDS = 4            # FOLLOW, LEFT, STRAIGHT, RIGHT
MAX_ANGLE = 0.5            # rad — debe coincidir con el limite del controlador
SEED = 42

# Indices de comando (coinciden con cil_data_collector.CMD_*)
CMD_FOLLOW, CMD_LEFT, CMD_STRAIGHT, CMD_RIGHT = 0, 1, 2, 3
CMD_NAMES = {CMD_FOLLOW: "FOLLOW", CMD_LEFT: "LEFT",
             CMD_STRAIGHT: "STRAIGHT", CMD_RIGHT: "RIGHT"}
# Al voltear la imagen el comando direccional se invierte; los demas quedan igual.
CMD_FLIP = {CMD_FOLLOW: CMD_FOLLOW, CMD_LEFT: CMD_RIGHT,
            CMD_STRAIGHT: CMD_STRAIGHT, CMD_RIGHT: CMD_LEFT}

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")


# =============================================================================
# 1. CARGA Y PREPROCESAMIENTO DEL DATASET
# =============================================================================

def preprocess(bgr):
    """
    Preprocesa una imagen BGR cruda al formato de entrada del modelo:
      - recorta la franja util de la carretera (quita cielo y cofre),
      - convierte BGR -> RGB,
      - redimensiona a (IMG_H, IMG_W),
      - normaliza a [0, 1].
    El recorte usa el 40%-90% inferior de la altura, donde vive la carretera en la
    camara del BmwX5 (256x128); ese rango se mantiene identico en inferencia.
    """
    h = bgr.shape[0]
    crop = bgr[int(0.40 * h):int(0.90 * h), :, :]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def load_dataset(data_dir):
    """
    Lee driving_log.csv y carga (imagen, comando, angulo) para cada fila valida.
    Espera la estructura producida por el controlador de recoleccion:
        <data_dir>/driving_log.csv
        <data_dir>/IMG/<image_name>.png
    """
    csv_path = os.path.join(data_dir, "driving_log.csv")
    img_dir = os.path.join(data_dir, "IMG")
    X, C, y = [], [], []
    faltantes = 0

    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            img_path = os.path.join(img_dir, row["image_name"])
            bgr = cv2.imread(img_path)
            if bgr is None:
                faltantes += 1
                continue
            X.append(preprocess(bgr))
            C.append(int(row["command"]))
            y.append(float(row["steering_angle"]))

    X = np.asarray(X, dtype=np.float32)
    C = np.asarray(C, dtype=np.int64)
    y = np.asarray(y, dtype=np.float32)
    print(f"[DATA] Cargadas {len(X)} imagenes ({faltantes} faltantes/ilegibles)")
    _resumen_balance(C, "crudo")
    return X, C, y


def _resumen_balance(C, etiqueta):
    """Imprime cuantas muestras hay por comando (para vigilar el balance del dataset)."""
    conteo = {CMD_NAMES[i]: int(np.sum(C == i)) for i in range(NUM_COMMANDS)}
    print(f"[BALANCE] ({etiqueta}) {conteo}")


# =============================================================================
# 2. DATA AUGMENTATION ESPECIFICO DE BEHAVIORAL CLONING
# =============================================================================

def balance_dataset(X, C, y, follow_ratio=3.0, rng=None):
    """
    Acerca el balance entre comandos ANTES del augmentation:
      - submuestrea FOLLOW (que suele dominar) a `follow_ratio` veces la clase de
        giro mas numerosa,
      - sobremuestrea (con repeticion) LEFT / STRAIGHT / RIGHT hasta esa misma clase.
    Evita que el modelo se sesgue a ir siempre recto cuando el dataset tiene pocos
    giros. Retorna los arreglos balanceados.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    counts = {k: int(np.sum(C == k)) for k in range(NUM_COMMANDS)}
    turnos = [counts[k] for k in (CMD_LEFT, CMD_STRAIGHT, CMD_RIGHT) if counts[k] > 0]
    if not turnos:
        print("[BALANCE] No hay giros; se omite el rebalanceo.")
        return X, C, y
    objetivo = max(turnos)
    follow_target = int(objetivo * follow_ratio)

    seleccion = []
    for k in range(NUM_COMMANDS):
        ids = np.where(C == k)[0]
        if len(ids) == 0:
            continue
        if k == CMD_FOLLOW:
            sel = rng.choice(ids, min(len(ids), follow_target), replace=False)
        else:
            sel = rng.choice(ids, objetivo, replace=len(ids) < objetivo)
        seleccion.append(sel)
    idx = np.concatenate(seleccion)
    rng.shuffle(idx)
    print(f"[BALANCE] {len(X)} -> {len(idx)} muestras (FOLLOW<= {follow_target}, "
          f"cada giro -> {objetivo})")
    _resumen_balance(C[idx], "balanceado")
    return X[idx], C[idx], y[idx]


def augment(X, C, y, brightness_copies=1, rng=None):
    """
    Expande el dataset con variantes que preservan la semantica de la conduccion:

      - FLIP HORIZONTAL: por cada muestra se agrega su espejo con el angulo
        negado y el comando direccional intercambiado (LEFT<->RIGHT). Esto
        DUPLICA el dataset y lo balancea entre giros a izquierda y derecha; es
        la tecnica estandar en clonado de comportamiento (a diferencia de GTSRB,
        donde el flip esta prohibido porque invierte el significado de la senal).

      - JITTER DE BRILLO: por cada muestra (y su flip) se agregan
        `brightness_copies` copias con ganancia de brillo aleatoria, para que el
        modelo generalice ante cambios de iluminacion entre el Mundo #1 y el #2.

    Retorna los arreglos aumentados (imagen, comando, angulo).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    Xa, Ca, ya = [X], [C], [y]

    # --- Flip horizontal ---
    X_flip = X[:, :, ::-1, :].copy()                    # espejo en el eje del ancho
    y_flip = -y                                         # angulo negado
    C_flip = np.array([CMD_FLIP[int(c)] for c in C], dtype=C.dtype)
    Xa.append(X_flip); Ca.append(C_flip); ya.append(y_flip)

    # --- Jitter de brillo (sobre originales + flips) ---
    base_X = np.concatenate([X, X_flip], axis=0)
    base_C = np.concatenate([C, C_flip], axis=0)
    base_y = np.concatenate([y, y_flip], axis=0)
    for _ in range(brightness_copies):
        ganancia = rng.uniform(0.6, 1.4, size=(len(base_X), 1, 1, 1)).astype(np.float32)
        Xb = np.clip(base_X * ganancia, 0.0, 1.0)
        Xa.append(Xb); Ca.append(base_C.copy()); ya.append(base_y.copy())

    X_out = np.concatenate(Xa, axis=0)
    C_out = np.concatenate(Ca, axis=0)
    y_out = np.concatenate(ya, axis=0)

    # Mezcla para que el split train/val no quede ordenado por tipo de aumento.
    idx = rng.permutation(len(X_out))
    X_out, C_out, y_out = X_out[idx], C_out[idx], y_out[idx]
    print(f"[AUGMENT] Dataset: {len(X)} -> {len(X_out)} imagenes")
    _resumen_balance(C_out, "aumentado")
    return X_out, C_out, y_out


# =============================================================================
# 3. MODELO CIL RAMIFICADO
# =============================================================================

def construir_backbone(image_in):
    """
    Backbone convolucional ligero estilo Bojarski et al. (2016) — el modelo que
    recomienda el enunciado del proyecto. Cinco capas convolucionales (24/36/48/64/64)
    con reduccion espacial por stride, seguidas de un Flatten y una capa densa. Se
    conserva el Flatten (no GlobalAveragePooling) porque la POSICION del carril en la
    imagen es informacion clave para la regresion del angulo.

    Frente al backbone de 8 capas de Codevilla (~38M parametros), esta version tiene
    ~2-3M parametros, lo que permite entrenar en CPU en tiempos razonables sin perder
    calidad para esta tarea.
    """
    x = layers.Conv2D(24, 5, strides=2, activation="relu", padding="same")(image_in)
    x = layers.Conv2D(36, 5, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(48, 5, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    z = layers.Dense(128, activation="relu")(x)
    z = layers.Dropout(0.4)(z)
    return z


def construir_cil():
    r"""
    Modelo CIL ramificado con dos entradas y una salida:
       entradas: imagen (66,200,3) y comando one-hot (4,)
       salida:   angulo de direccion de la rama seleccionada por el comando.

    Las cuatro ramas se apilan en un tensor (N,4); al multiplicarlo por el comando
    one-hot y sumar sobre el eje de comandos, solo sobrevive la prediccion de la rama
    activa. Esto mantiene UNA sola salida (MSE limpio) y un grafo exportable a TFLite.
    """
    image_in = Input(shape=(IMG_H, IMG_W, 3), name="image")
    cmd_in = Input(shape=(NUM_COMMANDS,), name="command")   # one-hot

    z = construir_backbone(image_in)

    ramas = []
    for i in range(NUM_COMMANDS):
        nombre = CMD_NAMES[i].lower()
        # Cabezal condicional por comando (ligero): FC + salida en tanh.
        b = layers.Dense(64, activation="relu", name=f"branch_{nombre}_fc")(z)
        b = layers.Dropout(0.3)(b)
        b = layers.Dense(1, activation="tanh", name=f"branch_{nombre}_out")(b)
        ramas.append(b)
    # tanh -> [-1,1]; se escala al rango fisico del volante (+/-MAX_ANGLE).
    todas = layers.Concatenate(name="branches")(ramas)            # (N, 4)
    todas = layers.Lambda(lambda t: t * MAX_ANGLE, name="scale")(todas)

    # Seleccion por mascara: (N,4) * (N,4) -> suma sobre comandos -> (N,1)
    seleccion = layers.Multiply(name="select")([todas, cmd_in])
    steering = layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=1, keepdims=True), name="steering"
    )(seleccion)

    model = models.Model(inputs=[image_in, cmd_in], outputs=steering, name="cil_branched")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


# =============================================================================
# 4. ENTRENAMIENTO Y EVALUACION
# =============================================================================

def entrenar(model, X, C, y, epochs, batch_size=64, val_split=0.2):
    """Entrena con split aleatorio y callbacks de early stopping / reduccion de LR."""
    C_oh = tf.keras.utils.to_categorical(C, NUM_COMMANDS).astype(np.float32)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_split)
    val, tr = idx[:n_val], idx[n_val:]

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    history = model.fit(
        [X[tr], C_oh[tr]], y[tr],
        validation_data=([X[val], C_oh[val]], y[val]),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=2,
    )
    return history, (X[val], C_oh[val], y[val], C[val])


def evaluar(model, val_data):
    """Reporta el MAE de steering global y desglosado por comando de navegacion."""
    Xv, Cv_oh, yv, Cv = val_data
    pred = model.predict([Xv, Cv_oh], verbose=0).ravel()
    mae_global = float(np.mean(np.abs(pred - yv)))
    print(f"[EVAL] MAE global de steering: {mae_global:.4f} rad")
    for i in range(NUM_COMMANDS):
        m = Cv == i
        if np.any(m):
            mae_i = float(np.mean(np.abs(pred[m] - yv[m])))
            print(f"[EVAL]   {CMD_NAMES[i]:8s}: MAE {mae_i:.4f} rad  (n={int(np.sum(m))})")
    return mae_global


# =============================================================================
# 5. EXPORTACION (Keras + TFLite) Y VERIFICACION DE PARIDAD
#    (patron reutilizado de train_gtsrb_cnn.py:331-364, adaptado a regresion)
# =============================================================================

def exportar(model, val_data=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    keras_path = os.path.join(MODEL_DIR, "cil_model.keras")
    tflite_path = os.path.join(MODEL_DIR, "cil_model.tflite")

    model.save(keras_path)
    print(f"[EXPORT] Modelo Keras guardado en {keras_path}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # El grafo usa tf.reduce_sum dentro de un Lambda: habilitar ops de TF de respaldo.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"[EXPORT] Modelo TFLite guardado en {tflite_path} "
          f"({len(tflite_model)/1024:.0f} KB)")

    # --- Paridad Keras vs TFLite sobre 20 muestras de validacion ---
    if val_data is not None:
        Xv, Cv_oh, yv, _ = val_data
        n = min(20, len(Xv))
        interp = tf.lite.Interpreter(model_content=tflite_model)
        interp.allocate_tensors()
        in_details = {d["name"]: d for d in interp.get_input_details()}
        out_det = interp.get_output_details()[0]
        # Los nombres de entrada pueden variar; se mapean por forma (imagen=4D, comando=2D).
        img_in = next(d for d in in_details.values() if len(d["shape"]) == 4)
        cmd_in = next(d for d in in_details.values() if len(d["shape"]) == 2)

        keras_pred = model.predict([Xv[:n], Cv_oh[:n]], verbose=0).ravel()
        tfl_pred = []
        for i in range(n):
            interp.set_tensor(img_in["index"], Xv[i:i+1].astype(np.float32))
            interp.set_tensor(cmd_in["index"], Cv_oh[i:i+1].astype(np.float32))
            interp.invoke()
            tfl_pred.append(float(interp.get_tensor(out_det["index"]).ravel()[0]))
        dif = float(np.max(np.abs(keras_pred - np.array(tfl_pred))))
        print(f"[EXPORT] Paridad Keras vs TFLite ({n} muestras): "
              f"diferencia maxima {dif:.6f} rad")
    return keras_path, tflite_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Entrenamiento del modelo CIL ramificado")
    ap.add_argument("--data_dir", default="./cil_dataset",
                    help="Carpeta con driving_log.csv e IMG/")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--brightness_copies", type=int, default=1,
                    help="Copias con jitter de brillo por muestra (controla el tamano final)")
    ap.add_argument("--balance", action="store_true",
                    help="Rebalancear comandos (submuestrea FOLLOW, sobremuestrea giros)")
    ap.add_argument("--follow_ratio", type=float, default=3.0,
                    help="Cuantas veces FOLLOW respecto a la clase de giro mas numerosa")
    args = ap.parse_args()

    print("=" * 70)
    print("  Entrenamiento CIL ramificado (Proyecto Final MR4010.10)")
    print("=" * 70)
    print(f"TensorFlow {tf.__version__} | GPU: {tf.config.list_physical_devices('GPU')}")

    X, C, y = load_dataset(args.data_dir)
    if args.balance:
        X, C, y = balance_dataset(X, C, y, follow_ratio=args.follow_ratio)
    X, C, y = augment(X, C, y, brightness_copies=args.brightness_copies)

    model = construir_cil()
    model.summary()
    history, val_data = entrenar(model, X, C, y,
                                 epochs=args.epochs, batch_size=args.batch_size)
    evaluar(model, val_data)
    exportar(model, val_data)
    print("[OK] Entrenamiento y exportacion completados.")


if __name__ == "__main__":
    main()
