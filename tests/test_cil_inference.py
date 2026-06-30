"""
===============================================================================
  Proyecto Final — CIL  ·  Pruebas automatizadas de inferencia
===============================================================================

  Estas pruebas verifican, sin necesidad de Webots, que:

  1. (siempre) La logica de seleccion por mascara one-hot del modelo CIL escoge
     correctamente la salida de la rama correspondiente al comando activo. Es la
     pieza que hace "condicional" al Imitation Learning, asi que se valida de forma
     aislada con NumPy puro (corre en cualquier entorno).

  2. (si hay TensorFlow + OpenCV) El modelo CIL exportado a TFLite responde, ante
     una imagen sintetica, con un angulo finito y dentro de [-MAX_ANGLE, MAX_ANGLE]
     para cada uno de los cuatro comandos, y que las ramas NO son identicas entre si
     (confirma que el ramificado realmente discrimina por comando). Si no hay un
     modelo entrenado, se construye uno sin entrenar solo para validar el grafo.

  Ejecucion:
      python tests/test_cil_inference.py        # imprime PASS / SKIP
      pytest tests/test_cil_inference.py         # como suite de pytest
===============================================================================
"""

import os
import sys
import tempfile
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..")
sys.path.append(os.path.join(REPO_ROOT, "cil_training"))

NUM_COMMANDS = 4
MAX_ANGLE = 0.5


# ---------------------------------------------------------------------------
# 1. Prueba pura (NumPy) — seleccion por mascara one-hot
# ---------------------------------------------------------------------------

def test_branch_selection_masking():
    """La salida seleccionada debe ser exactamente la de la rama del comando activo."""
    rng = np.random.default_rng(0)
    n = 50
    branches = rng.uniform(-MAX_ANGLE, MAX_ANGLE, size=(n, NUM_COMMANDS))
    cmds = rng.integers(0, NUM_COMMANDS, size=n)
    onehot = np.eye(NUM_COMMANDS)[cmds]

    # Replica de la operacion del modelo: (N,4)*(N,4) -> suma sobre comandos -> (N,)
    seleccion = np.sum(branches * onehot, axis=1)
    esperado = branches[np.arange(n), cmds]

    assert np.allclose(seleccion, esperado), "La mascara one-hot no selecciona la rama correcta"
    print("[PASS] Seleccion por mascara one-hot correcta")


# ---------------------------------------------------------------------------
# 2. Prueba de inferencia TFLite (se salta si faltan dependencias)
# ---------------------------------------------------------------------------

def _try_imports():
    try:
        import tensorflow as tf  # noqa: F401
        import cv2  # noqa: F401
        import train_cil  # noqa: F401
        return True
    except Exception as e:  # ImportError u otros
        print(f"[SKIP] Inferencia TFLite (faltan dependencias: {e})")
        return False


def _locate_or_build_tflite():
    """Devuelve la ruta a un cil_model.tflite: usa el entrenado si existe, o crea uno temporal."""
    import tensorflow as tf
    import train_cil

    for cand in [
        os.path.join(REPO_ROOT, "controllers", "cil_autonomous", "model", "cil_model.tflite"),
        os.path.join(REPO_ROOT, "cil_training", "model", "cil_model.tflite"),
    ]:
        if os.path.exists(cand):
            print(f"[INFO] Usando modelo entrenado: {cand}")
            return cand

    # No hay modelo entrenado: construir uno sin entrenar solo para validar el grafo.
    print("[INFO] Sin modelo entrenado; se construye uno temporal para validar el grafo.")
    model = train_cil.construir_cil()
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    blob = conv.convert()
    tmp = os.path.join(tempfile.gettempdir(), "cil_model_test.tflite")
    with open(tmp, "wb") as f:
        f.write(blob)
    return tmp


def test_tflite_inference():
    if not _try_imports():
        return
    import tensorflow as tf
    from train_cil import IMG_H, IMG_W

    path = _locate_or_build_tflite()
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    ins = interp.get_input_details()
    out = interp.get_output_details()[0]
    img_in = next(d for d in ins if len(d["shape"]) == 4)
    cmd_in = next(d for d in ins if len(d["shape"]) == 2)

    rng = np.random.default_rng(1)
    img = rng.random((1, IMG_H, IMG_W, 3), dtype=np.float32)

    angulos = []
    for c in range(NUM_COMMANDS):
        onehot = np.zeros((1, NUM_COMMANDS), dtype=np.float32)
        onehot[0, c] = 1.0
        interp.set_tensor(img_in["index"], img)
        interp.set_tensor(cmd_in["index"], onehot)
        interp.invoke()
        a = float(interp.get_tensor(out["index"]).ravel()[0])
        angulos.append(a)
        assert np.isfinite(a), f"Angulo no finito para el comando {c}"
        assert -MAX_ANGLE - 1e-4 <= a <= MAX_ANGLE + 1e-4, f"Angulo fuera de rango: {a}"

    # Las ramas deben discriminar: no todas las salidas pueden ser identicas.
    assert len(set(round(a, 6) for a in angulos)) > 1, \
        "Las 4 ramas devuelven el mismo angulo (el ramificado no discrimina)"
    print(f"[PASS] Inferencia TFLite OK; angulos por comando: "
          f"{[round(a, 4) for a in angulos]}")


if __name__ == "__main__":
    test_branch_selection_masking()
    test_tflite_inference()
    print("Pruebas finalizadas.")
