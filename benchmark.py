"""
================================================================
  Benchmark — K-Means Serial vs Paralelo (OpenMP)
  Proyecto Apertura — Cómputo Paralelo y en la Nube, ITAM 2026
================================================================
  Configura:
    Entrada: CSV externo (generado por synthetic_clusters.ipynb) o datos sintéticos
    Hilos   : {1, cores/2, cores, cores*2}
    Reps    : 10 por configuración (promediadas)
    Dims    : Detectadas del CSV o 2D/3D sintéticos | k=8 clusters

  Produce:
    results.csv            datos crudos
    speedup_2d.png         speedup en 2D
    speedup_3d.png         speedup en 3D
    speedup_combined.png   2D y 3D lado a lado

  Uso:
    python benchmark.py                                     (datos sintéticos por defecto)
    python benchmark.py --reps 5 --sizes 100000,200000    (5 reps, tamaños específicos)
    python benchmark.py --k 10 --max_iter 500 --seed 99   (10 clusters, 500 iter)
    python benchmark.py --input_csv data.csv              (usar CSV externo)
================================================================
"""

# Importa librerías estándar necesarias para manejo de argumentos,
# archivos CSV, sistema de archivos, ejecución de programas externos,
# temporización y paralelismo según CPU
import argparse
import csv
import os
import subprocess
import sys
import time
import multiprocessing

# Importa librerías de terceros para cálculo numérico y generación de datos
import numpy as np
from sklearn.datasets import make_blobs

try:
    # Matplotlib solo es necesario si se generan gráficos.
    import matplotlib
    matplotlib.use("Agg")  # Uso de backend sin GUI para scripts
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_PLOT = True
except ImportError:
    # Si matplotlib no está instalado, se continúa sin las gráficas.
    HAS_PLOT = False
    print("[WARN] matplotlib no disponible — se omiten graficas")


# ─────────────────────────────────────────────
#  Ensure using virtual environment
# ─────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
if sys.platform == 'win32':
    venv_python = os.path.join(script_dir, '.venv', 'Scripts', 'python.exe')
else:
    venv_python = os.path.join(script_dir, '.venv', 'bin', 'python')

if sys.executable != venv_python:
    subprocess.run([venv_python, __file__] + sys.argv[1:])
    sys.exit()


# ─────────────────────────────────────────────
#  Parametros globales (pueden sobreescribirse via CLI)
# ─────────────────────────────────────────────
# Configuración por defecto del benchmark
POINT_SIZES  = [100_000, 200_000, 300_000, 400_000, 600_000, 800_000, 1_000_000]
K            = 8      # número de clusters a usar
MAX_ITER     = 300    # iteraciones máximas por ejecución de k-means
REPS         = 10     # número de repeticiones por configuración (promediadas)
SEED         = 42     # semilla determinista para reproducción

# Rutas de binarios. En Windows el .exe se requiere.
SERIAL_BIN   = "./kmeans_serial.exe"
PARALLEL_BIN = "./kmeans_parallel.exe"

DATA_DIR     = "./data"       # carpeta para datasets generados
RESULTS_FILE = "results.csv"  # salida resumen de benchmark

# Colores y marcadores para gráficas de matplotlib
COLORS  = ["#1565C0", "#2E7D32", "#C62828", "#6A1B9A",
           "#E65100", "#00695C", "#AD1457", "#37474F"]
MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]


# ─────────────────────────────────────────────
#  Deteccion de cores
# ─────────────────────────────────────────────
def get_thread_configs():
    # Detecta cantidad de núcleos físicos/logicos del CPU y define configuraciones
    # de hilos para pruebas. Luego devuelve lista de hilos y núcleos.
    cores = multiprocessing.cpu_count()
    configs = sorted(set([1, max(1, cores // 2), cores, cores * 2]))
    print(f"[Info] Cores detectados: {cores}  -> hilos: {configs}")
    return configs, cores


# ─────────────────────────────────────────────
#  Generacion de datos
# ─────────────────────────────────────────────
def load_external_csv(path):
    # Carga un CSV externo (generado por synthetic_clusters.ipynb),
    # detecta número de puntos (filas) y dimensión (columnas).
    if not os.path.isfile(path):
        print(f"[ERROR] Archivo CSV no encontrado: {path}")
        sys.exit(1)

    with open(path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("[ERROR] CSV vacío")
        sys.exit(1)

    # Asumir que todas las filas tienen la misma cantidad de columnas
    dim = len(rows[0])
    n_points = len(rows)

    print(f"[IO] CSV externo: {n_points} puntos, dim={dim}")
    return n_points, dim, path


def generate_csv(n_points, dims, seed=SEED):
    # Crea carpeta de datos si no existe e intenta reutilizar archivos existentes
    os.makedirs(DATA_DIR, exist_ok=True)
    path = f"{DATA_DIR}/{n_points}_{dims}d.csv"
    if os.path.exists(path):
        return path

    print(f"\n  Generando {n_points:,} puntos {dims}D ...", end=" ", flush=True)

    # Genera puntos con clusters controlados
    pts = make_blobs(n_samples=n_points, centers=K, n_features=dims,
                     cluster_std=0.04, random_state=seed, center_box=(0, 1.0))[0]

    # Siempre valores positivos y redondeados para consistencia en datos
    pts = np.round(np.abs(pts), 3)
    np.savetxt(path, pts, delimiter=",", fmt="%.3f")

    print(f"listo -> {path}")
    return path


# ─────────────────────────────────────────────
#  Ejecucion de binarios
# ─────────────────────────────────────────────
def run_binary(cmd):
    # Ejecuta un comando externo (kmeans_serial.exe / kmeans_parallel.exe)
    # y mide el tiempo de ejecución.
    t0 = time.perf_counter()
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    elapsed = time.perf_counter() - t0

    # Si hubo error en el proceso, se informa y se devuelve NaN
    if res.returncode != 0:
        print(f"\n[ERROR] {' '.join(cmd)}\n{res.stderr.decode()}")
        return float("nan")
    return elapsed


def run_serial(csv_path):
    # Ejecuta el binario serial con un único set de parámetros:
    # - dataset CSV
    # - k clusters
    # - max_iter
    # - seed
    # Se devuelve el tiempo consumido en segundos.
    return run_binary([SERIAL_BIN, csv_path, str(K), str(MAX_ITER), str(SEED)])


def run_parallel(csv_path, threads):
    # Ejecuta el binario paralelo con el número de hilos deseado.
    # Comparado al serial, este invoca un proceso OpenMP que usa `threads`.
    return run_binary([PARALLEL_BIN, csv_path, str(K), str(threads), str(MAX_ITER), str(SEED)])


# ─────────────────────────────────────────────
#  Experimento
# ─────────────────────────────────────────────
def run_experiment(reps, thread_configs, dims, n_sizes, input_csv):
    # Ejecuta todo el conjunto de experimentos para las dimensiones y tamaños dados,
    # con cada configuración de hilos dada.
    records = []

    for dim in dims:
        print(f"\n{'='*55}")
        print(f"  Dimension: {dim}D")
        print(f"{'='*55}")

        for n in n_sizes:
            if input_csv:
                # Usar CSV externo directamente
                path = input_csv
                print(f"\n  n = {n:,}  dimension = {dim}D  (CSV externo)")
            else:
                # Generar datos sintéticos
                path = generate_csv(n, dim)
                print(f"\n  n = {n:,}  dimension = {dim}D")

            # 1) Modo serial
            print(f"    Serial            ({reps} reps) ...", end=" ", flush=True)
            st = [run_serial(path) for _ in range(reps)]  # repeticiones para promedio
            t_serial = float(np.nanmean(st))
            print(f"media = {t_serial:.3f}s  std = {np.nanstd(st):.3f}s")
            records.append(dict(
                dim=dim, n=n, version="serial", threads=1,
                t_mean=t_serial, t_std=float(np.nanstd(st)),
                speedup=1.0, raw=st
            ))

            # Paralelo
            for threads in thread_configs:
                print(f"    Paralelo {threads:>2} hilos ({reps} reps) ...",
                      end=" ", flush=True)
                pt = [run_parallel(path, threads) for _ in range(reps)]
                t_par = float(np.nanmean(pt))
                sp = t_serial / t_par if t_par > 0 else float("nan")
                print(f"media = {t_par:.3f}s  speedup = {sp:.2f}x")
                records.append(dict(
                    dim=dim, n=n, version="parallel", threads=threads,
                    t_mean=t_par, t_std=float(np.nanstd(pt)),
                    speedup=sp, raw=pt
                ))

    return records


# ─────────────────────────────────────────────
#  Guardar CSV
# ─────────────────────────────────────────────
def save_results(records, reps):
    # Guarda un archivo CSV con todos los resultados del benchmark,
    # incluyendo los tiempos de cada repetición (rep_1..rep_n).
    rep_cols = [f"rep_{i+1}" for i in range(reps)]
    with open(RESULTS_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dim", "n_points", "version", "threads",
                    "time_mean_s", "time_std_s", "speedup"] + rep_cols)
        for r in records:
            # Si la carrera devolvió menos repeticiones, completar con NaN
            raw = r["raw"] + [float("nan")] * max(0, reps - len(r["raw"]))
            w.writerow([
                r["dim"], r["n"], r["version"], r["threads"],
                round(r["t_mean"], 5), round(r["t_std"], 5),
                round(r["speedup"], 4)
            ] + [round(t, 5) for t in raw[:reps]])
    print(f"\n[IO] {RESULTS_FILE}")


# ─────────────────────────────────────────────
#  Graficas
# ─────────────────────────────────────────────
def get_speedup_curve(records, dim, threads):
    # Construye un conjunto de puntos (x,y,error) para la curva de speedup:
    # x = tamaño en millones de puntos, y = speedup relativo frente al serial.
    xs, ys, errs = [], [], []
    for n in sorted(set(r["n"] for r in records)):
        s = next((r for r in records if r["dim"]==dim and r["n"]==n
                  and r["version"]=="serial"), None)
        p = next((r for r in records if r["dim"]==dim and r["n"]==n
                  and r["version"]=="parallel" and r["threads"]==threads), None)
        if s and p and p["t_mean"] > 0 and not np.isnan(p["t_mean"]):
            sp   = s["t_mean"] / p["t_mean"]
            # Error de propagación de la división (asumiendo incertidumbres de t_std)
            rerr = np.sqrt((s["t_std"]/s["t_mean"])**2 + (p["t_std"]/p["t_mean"])**2) * sp
            xs.append(n / 1_000_000)
            ys.append(sp)
            errs.append(rerr)
    return xs, ys, errs


def plot_speedup(records, dim, ax, thread_configs):
    # Dibuja la curva de speedup para cada configuración de hilos en un eje dado.
    # - xs: tamaño de dataset en millones
    # - ys: speedup relativo
    # - errs: barras de error (propagación)
    for i, threads in enumerate(thread_configs):
        xs, ys, errs = get_speedup_curve(records, dim, threads)
        if xs:
            ax.errorbar(xs, ys, yerr=errs,
                        label=f"{threads} hilo{'s' if threads>1 else ''}",
                        color=COLORS[i % len(COLORS)],
                        marker=MARKERS[i % len(MARKERS)],
                        linewidth=2, markersize=7, capsize=4)

    # Línea de referencia de speedup 1.5 (puede ser utilidad visual)
    ax.axhline(1.5, color="red", linestyle="--", linewidth=1.5, alpha=0.8,
               label="Minimo requerido (1.5x)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.4)
    ax.set_xlabel("Puntos (millones)", fontsize=11)
    ax.set_ylabel("Speedup  (T_serial / T_paralelo)", fontsize=11)
    ax.set_title(f"Speedup K-Means OpenMP — {dim}D  (k={K})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1fM"))


def plot_times(records, dim, ax, thread_configs):
    # Grafica los tiempos de ejecución (tiempo medio) de serial y cada paralelo.
    xs = sorted(set(r["n"] for r in records))
    xp = [n / 1_000_000 for n in xs]  # eje x en millones de puntos

    ys = [next((r["t_mean"] for r in records if r["dim"]==dim and r["n"]==n
                and r["version"]=="serial"), np.nan) for n in xs]
    ax.plot(xp, ys, "k--o", linewidth=2, markersize=6, label="Serial", zorder=5)

    for i, threads in enumerate(thread_configs):
        yp = [next((r["t_mean"] for r in records if r["dim"]==dim and r["n"]==n
                    and r["version"]=="parallel" and r["threads"]==threads), np.nan)
              for n in xs]
        ax.plot(xp, yp, color=COLORS[i%len(COLORS)], marker=MARKERS[i%len(MARKERS)],
                linewidth=2, markersize=6,
                label=f"{threads} hilo{'s' if threads>1 else ''}")

    ax.set_xlabel("Puntos (millones)", fontsize=11)
    ax.set_ylabel("Tiempo (s)", fontsize=11)
    ax.set_title(f"Tiempos de Ejecucion — {dim}D  (k={K})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1fM"))


def generate_plots(records, thread_configs):
    # Genera gráficos de speedup y tiempos si matplotlib está disponible.
    if not HAS_PLOT:
        return
    for dim in [2, 3]:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Evaluacion de Desempeno — K-Means con OpenMP ({dim}D)\n"
            f"k={K}, 10 repeticiones por configuracion",
            fontsize=13, fontweight="bold"
        )
        plot_speedup(records, dim, a1, thread_configs)
        plot_times(records, dim, a2, thread_configs)
        plt.tight_layout()
        fname = f"speedup_{dim}d.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Plot] {fname}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Speedup K-Means Paralelo (OpenMP) — 2D vs 3D\n"
        f"k={K}, 10 repeticiones por configuracion",
        fontsize=13, fontweight="bold"
    )
    plot_speedup(records, 2, a1, thread_configs)
    plot_speedup(records, 3, a2, thread_configs)
    a2.set_ylabel("")
    plt.tight_layout()
    plt.savefig("speedup_combined.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[Plot] speedup_combined.png")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def parse_args():
    # Define argumentos de línea de comandos:
    # --k: número de clusters (default: K)
    # --reps: número de repeticiones por configuración (default: REPS)
    # --max_iter: iteraciones máximas de k-means (default: MAX_ITER)
    # --seed: semilla para reproducibilidad (default: SEED)
    # --sizes: lista de tamaños separados por coma (solo si no hay --input_csv)
    # --input_csv: archivo CSV de entrada (generado por synthetic_clusters.ipynb)
    p = argparse.ArgumentParser(description="K-Means Benchmark Serial vs Paralelo (OpenMP)")
    p.add_argument("--k",    type=int, default=K,
                   help=f"Número de clusters (default: {K})")
    p.add_argument("--reps", type=int, default=REPS,
                   help=f"Número de repeticiones por configuración (default: {REPS})")
    p.add_argument("--max_iter", type=int, default=MAX_ITER,
                   help=f"Iteraciones máximas de k-means (default: {MAX_ITER})")
    p.add_argument("--seed", type=int, default=SEED,
                   help=f"Semilla para reproducibilidad (default: {SEED})")
    p.add_argument("--sizes", type=str, default=None,
                   help="Tamanios separados por coma, ej: 100000,500000 (ignorado si --input_csv)")
    p.add_argument("--input_csv", type=str, default=None,
                   help="Archivo CSV de entrada (generado por synthetic_clusters.ipynb)")
    return p.parse_args()


def main():
    global K, POINT_SIZES, MAX_ITER, REPS, SEED
    args = parse_args()
    K = args.k
    MAX_ITER = args.max_iter
    REPS = args.reps
    SEED = args.seed

    input_csv_path = None
    detected_dims = [2, 3]  # por defecto
    detected_n = POINT_SIZES  # por defecto

    if args.input_csv:
        # Cargar CSV externo y detectar parámetros
        n_points, dim, csv_path = load_external_csv(args.input_csv)
        input_csv_path = csv_path
        detected_dims = [dim]
        detected_n = [n_points]
        print(f"[Info] Usando CSV externo: {args.input_csv} (n={n_points}, dim={dim})")
    else:
        # Comportamiento original: generar datos
        if args.sizes:
            POINT_SIZES = [int(s) for s in args.sizes.split(",")]
        detected_n = POINT_SIZES

    # Verifica que los ejecutables existan antes de comenzar
    for b in [SERIAL_BIN, PARALLEL_BIN]:
        if not os.path.isfile(b):
            print(f"[ERROR] Binario no encontrado: {b}")
            print("  Compila con:")
            print("    g++ -O2 -std=c++17 -o kmeans_serial kmeans_serial.cpp")
            print("    g++ -O2 -std=c++17 -fopenmp -o kmeans_parallel kmeans_parallel.cpp")
            sys.exit(1)

    # Configuración de hilos según CPU
    thread_configs, cores = get_thread_configs()

    # Impresión de parámetros de benchmark
    print(f"\n{'='*55}")
    print(f"  Benchmark  k = {K}  reps = {args.reps}  cores = {cores}")
    print(f"  max_iter = {MAX_ITER}  seed = {SEED}")
    if input_csv_path:
        print(f"  CSV Entrada: {os.path.basename(input_csv_path)}")
    else:
        print(f"  Tamaños: {[f'{n:,}' for n in detected_n]}")
    print(f"  Dimensiones: {detected_dims}")
    print(f"  Hilos:    {thread_configs}")
    print(f"{'='*55}")

    # Ejecutar experimentos, guardar resultados y gráficas
    records = run_experiment(reps=args.reps, thread_configs=thread_configs,
                             dims=detected_dims, n_sizes=detected_n, input_csv=input_csv_path)
    save_results(records, reps=args.reps)
    generate_plots(records, thread_configs)

    # Resumen de speedup por dimensión y cantidad de hilos
    print(f"\n{'='*55}")
    print("  RESUMEN DE SPEEDUPS")
    print(f"{'='*55}")
    for dim in detected_dims:
        print(f"\n  {dim}D:")
        for threads in thread_configs:
            sp_vals = [r["speedup"] for r in records
                       if r["dim"]==dim and r["version"]=="parallel"
                       and r["threads"]==threads and not np.isnan(r["speedup"])]
            if sp_vals:
                print(f"    {threads:>2} hilos -> max = {max(sp_vals):.2f}x  "
                      f"media = {np.mean(sp_vals):.2f}x")


if __name__ == "__main__":
    main()
