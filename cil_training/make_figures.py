"""
===============================================================================
  Proyecto Final — CIL  ·  Generacion de todas las figuras del reporte
===============================================================================

  Reproduce las visualizaciones del notebook y las guarda como PNG en
  ../screenshots/, para incluirlas en el reporte y el video:

    01 balance por comando        07 balance tras rebalanceo
    02 histograma de angulo       08 curvas de entrenamiento
    03 angulo por comando         09 MAE por comando
    04 muestras por comando       10 prediccion vs real
    05 preprocesamiento           11 histograma de error
    06 data augmentation          12 predicciones de muestra

  Uso (con el env que tenga tensorflow + opencv + matplotlib):
      python make_figures.py --data_dir ../../cil_dataset
===============================================================================
"""

import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

import train_cil as cil

CMD_NAMES = cil.CMD_NAMES
NAMES = [CMD_NAMES[i] for i in range(cil.NUM_COMMANDS)]
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, "..", "screenshots")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIG] {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(SCRIPT_DIR, "..", "..", "cil_dataset"))
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    X, C, y = cil.load_dataset(args.data_dir)

    # 01 Balance por comando
    vals = [int(np.sum(C == k)) for k in range(cil.NUM_COMMANDS)]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars = ax.bar(NAMES, vals, color=COLORS)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom")
    ax.set_title("Muestras por comando de navegacion"); ax.set_ylabel("# imagenes")
    save(fig, "01_balance_comando.png")

    # 02 Histograma de angulo
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.hist(y, bins=41, color="#4C72B0", edgecolor="white")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_title("Distribucion del angulo de direccion"); ax.set_xlabel("angulo (rad)")
    ax.set_ylabel("frecuencia")
    save(fig, "02_hist_angulo.png")

    # 03 Angulo por comando (boxplot)
    data = [(NAMES[i], y[C == i]) for i in range(cil.NUM_COMMANDS) if np.any(C == i)]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.boxplot([d for _, d in data], labels=[n for n, _ in data], showmeans=True)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.set_title("Angulo de direccion por comando"); ax.set_ylabel("angulo (rad)")
    save(fig, "03_angulo_por_comando.png")

    # 04 Muestras por comando
    cmds = [i for i in range(cil.NUM_COMMANDS) if np.any(C == i)]
    fig, axes = plt.subplots(len(cmds), 5, figsize=(5 * 2.1, len(cmds) * 1.7))
    axes = np.atleast_2d(axes)
    for r, ci in enumerate(cmds):
        idx = np.where(C == ci)[0]
        sel = idx[np.linspace(0, len(idx) - 1, 5).astype(int)]
        for c in range(5):
            a = axes[r][c]; a.imshow(X[sel[c]]); a.set_xticks([]); a.set_yticks([])
            if c == 0:
                a.set_ylabel(NAMES[ci])
    fig.suptitle("Muestras de la camara por comando")
    save(fig, "04_muestras_por_comando.png")

    # 05 Preprocesamiento
    raw_path = sorted(glob.glob(os.path.join(args.data_dir, "IMG", "*.*")))[0]
    raw = cv2.cvtColor(cv2.imread(raw_path), cv2.COLOR_BGR2RGB)
    proc = cil.preprocess(cv2.imread(raw_path))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    ax[0].imshow(raw); ax[0].set_title(f"Original {raw.shape[1]}x{raw.shape[0]}"); ax[0].axis("off")
    ax[1].imshow(proc); ax[1].set_title(f"Preprocesada {cil.IMG_W}x{cil.IMG_H}"); ax[1].axis("off")
    save(fig, "05_preprocesamiento.png")

    # 06 Data augmentation
    cand = np.where(np.abs(y) > 0.05)[0]
    i = int(cand[0]) if len(cand) else 0
    img, ang, cmd = X[i], float(y[i]), int(C[i])
    flip = img[:, ::-1, :]; ang_f = -ang; cmd_f = cil.CMD_FLIP[cmd]
    vistas = [(img, f"original\ncmd={CMD_NAMES[cmd]} ang={ang:+.2f}"),
              (flip, f"flip\ncmd={CMD_NAMES[cmd_f]} ang={ang_f:+.2f}"),
              (np.clip(img * 1.4, 0, 1), "brillo +40%"),
              (np.clip(img * 0.6, 0, 1), "brillo -40%")]
    fig, ax = plt.subplots(1, 4, figsize=(13, 2.8))
    for a, (im, t) in zip(ax, vistas):
        a.imshow(im); a.set_title(t, fontsize=9); a.axis("off")
    fig.suptitle("Data augmentation: el flip niega el angulo e intercambia LEFT<->RIGHT")
    save(fig, "06_augmentation.png")

    # --- Rebalanceo + augmentation ---
    Xb, Cb, yb = cil.balance_dataset(X, C, y, follow_ratio=1.0)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(NAMES, [int(np.sum(Cb == k)) for k in range(cil.NUM_COMMANDS)], color=COLORS)
    ax.set_title("Balance tras el rebalanceo"); ax.set_ylabel("# imagenes")
    save(fig, "07_balance_rebalanceado.png")
    Xa, Ca, ya = cil.augment(Xb, Cb, yb, brightness_copies=0)

    # --- Entrenamiento (para las curvas) ---
    model = cil.construir_cil()
    history, val_data = cil.entrenar(model, Xa, Ca, ya, epochs=args.epochs, batch_size=128)

    # 08 Curvas
    h = history.history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(h["loss"], label="train"); ax[0].plot(h["val_loss"], label="val")
    ax[0].set_title("Perdida (MSE)"); ax[0].set_xlabel("epoca"); ax[0].legend()
    ax[1].plot(h["mae"], label="train"); ax[1].plot(h["val_mae"], label="val")
    ax[1].set_title("MAE de steering"); ax[1].set_xlabel("epoca"); ax[1].set_ylabel("rad"); ax[1].legend()
    save(fig, "08_curvas_entrenamiento.png")

    # --- Evaluacion ---
    Xv, Cv_oh, yv, Cv = val_data
    pred = model.predict([Xv, Cv_oh], verbose=0).ravel()

    # 09 MAE por comando
    mae_g = float(np.mean(np.abs(pred - yv)))
    maes = [float(np.mean(np.abs(pred[Cv == i] - yv[Cv == i]))) if np.any(Cv == i) else 0
            for i in range(cil.NUM_COMMANDS)]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars = ax.bar(NAMES, maes, color=COLORS)
    for b, v in zip(bars, maes):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
    ax.axhline(mae_g, color="k", ls="--", lw=1, label=f"global={mae_g:.3f}")
    ax.set_title("MAE del angulo por comando"); ax.set_ylabel("MAE (rad)"); ax.legend()
    save(fig, "09_mae_por_comando.png")

    # 10 Prediccion vs real
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for i in range(cil.NUM_COMMANDS):
        m = Cv == i
        if np.any(m):
            ax.scatter(yv[m], pred[m], s=8, alpha=0.4, color=COLORS[i], label=CMD_NAMES[i])
    lim = cil.MAX_ANGLE * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax.set_xlabel("real (rad)"); ax.set_ylabel("predicho (rad)")
    ax.set_title("Prediccion vs real"); ax.legend()
    save(fig, "10_pred_vs_real.png")

    # 11 Histograma de error
    err = pred - yv
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.hist(err, bins=41, color="#55A868", edgecolor="white")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_title(f"Error de prediccion (media {err.mean():+.3f}, sd {err.std():.3f})")
    ax.set_xlabel("error (rad)"); ax.set_ylabel("frecuencia")
    save(fig, "11_hist_error.png")

    # 12 Predicciones de muestra
    idx = np.random.default_rng(0).choice(len(Xv), size=min(6, len(Xv)), replace=False)
    fig, axes = plt.subplots(2, len(idx), figsize=(len(idx) * 2.1, 4),
                             gridspec_kw={"height_ratios": [3, 1]})
    for c, k in enumerate(idx):
        axes[0][c].imshow(Xv[k]); axes[0][c].axis("off")
        axes[0][c].set_title(f"{CMD_NAMES[int(Cv[k])]}\nreal {yv[k]:+.2f} / pred {pred[k]:+.2f}", fontsize=8)
        for val, col in [(yv[k], "#999999"), (pred[k], "#C44E52")]:
            axes[1][c].barh([0], [val], color=col, height=0.5)
        axes[1][c].set_xlim(-cil.MAX_ANGLE, cil.MAX_ANGLE); axes[1][c].axvline(0, color="k", lw=0.8)
        axes[1][c].set_yticks([]); axes[1][c].set_xticks([])
    fig.suptitle("Real (gris) vs predicho (rojo)")
    save(fig, "12_predicciones_muestra.png")

    print(f"\n[OK] Figuras guardadas en {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
