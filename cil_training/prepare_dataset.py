"""
===============================================================================
  Proyecto Final — CIL  ·  Preparacion del dataset para subir a GitHub
===============================================================================

  El dataset crudo del colector (PNG a 320x160) pesa mucho (~59 KB/imagen). Este
  script lo **comprime** hacia el repositorio `cil_dataset`:
    - reduce la resolucion (ancho objetivo, manteniendo proporcion),
    - guarda en JPG (mucho mas ligero que PNG para fotos),
    - opcionalmente rebalancea (submuestrea FOLLOW) para reducir aun mas el tamano.

  El recorte y el resize finales a la entrada del modelo (200x88) los sigue haciendo
  `train_cil.preprocess` sobre estas imagenes, asi que el entrenamiento sigue
  coincidiendo con la inferencia en Webots (que parte de la camara 320x160).

  Uso (con el Python que tenga opencv, p.ej. tu env de Webots):
      python prepare_dataset.py                       # comprime todo
      python prepare_dataset.py --balance             # ademas rebalancea FOLLOW
      python prepare_dataset.py --width 256 --quality 85
===============================================================================
"""

import os
import csv
import argparse
import numpy as np
import cv2

CMD_LEFT, CMD_STRAIGHT, CMD_RIGHT = 1, 2, 3
SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(SCRIPT_DIR, "..", "controllers", "cil_data_collector", "dataset")
DEFAULT_DST = os.path.join(SCRIPT_DIR, "..", "..", "cil_dataset")


def main():
    ap = argparse.ArgumentParser(description="Comprime el dataset para subir a GitHub")
    ap.add_argument("--src", default=DEFAULT_SRC, help="Carpeta del dataset crudo (con IMG/ y driving_log.csv)")
    ap.add_argument("--dst", default=DEFAULT_DST, help="Carpeta destino (repo cil_dataset)")
    ap.add_argument("--width", type=int, default=256, help="Ancho objetivo (mantiene proporcion)")
    ap.add_argument("--quality", type=int, default=85, help="Calidad JPG (1-100)")
    ap.add_argument("--balance", action="store_true", help="Submuestrear FOLLOW")
    ap.add_argument("--follow_ratio", type=float, default=3.0,
                    help="FOLLOW respecto a la clase de giro mas numerosa (con --balance)")
    args = ap.parse_args()

    src_csv = os.path.join(args.src, "driving_log.csv")
    src_img = os.path.join(args.src, "IMG")
    dst_img = os.path.join(args.dst, "IMG")
    dst_csv = os.path.join(args.dst, "driving_log.csv")
    os.makedirs(dst_img, exist_ok=True)

    rows = list(csv.DictReader(open(src_csv)))
    print(f"[SRC] {len(rows)} filas en {src_csv}")

    # --- Rebalanceo opcional (submuestrea FOLLOW) ---
    if args.balance:
        rng = np.random.default_rng(SEED)
        by_cmd = {}
        for r in rows:
            by_cmd.setdefault(int(r["command"]), []).append(r)
        turnos = [len(by_cmd.get(k, [])) for k in (CMD_LEFT, CMD_STRAIGHT, CMD_RIGHT) if by_cmd.get(k)]
        follow_cap = int(max(turnos) * args.follow_ratio) if turnos else len(rows)
        seleccion = []
        for k, lst in by_cmd.items():
            if k == 0 and len(lst) > follow_cap:
                idx = rng.choice(len(lst), follow_cap, replace=False)
                seleccion += [lst[i] for i in idx]
            else:
                seleccion += lst
        rows = seleccion
        print(f"[BALANCE] FOLLOW limitado a {follow_cap}; total ahora {len(rows)}")

    # --- Comprimir imagenes ---
    out_rows = []
    faltan = 0
    for i, r in enumerate(rows):
        bgr = cv2.imread(os.path.join(src_img, r["image_name"]))
        if bgr is None:
            faltan += 1
            continue
        h, w = bgr.shape[:2]
        if w > args.width:
            nh = int(h * args.width / w)
            bgr = cv2.resize(bgr, (args.width, nh), interpolation=cv2.INTER_AREA)
        base = os.path.splitext(r["image_name"])[0] + ".jpg"
        cv2.imwrite(os.path.join(dst_img, base),
                    bgr, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        r2 = dict(r); r2["image_name"] = base
        out_rows.append(r2)
        if (i + 1) % 2000 == 0:
            print(f"  ... {i+1}/{len(rows)}")

    with open(dst_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        wr.writeheader()
        wr.writerows(out_rows)

    total_mb = sum(os.path.getsize(os.path.join(dst_img, r["image_name"]))
                   for r in out_rows) / 1e6
    print(f"[OK] {len(out_rows)} imagenes escritas en {dst_img} "
          f"({faltan} ilegibles).  Peso IMG: {total_mb:.0f} MB")
    print(f"[OK] CSV: {dst_csv}")
    print("Ahora: cd al repo cil_dataset, git add -A, commit y push.")


if __name__ == "__main__":
    main()
