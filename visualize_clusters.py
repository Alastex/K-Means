"""
================================================================
  Visualizacion de Clusters — K-Means con OpenMP
  Cómputo Paralelo y en la Nube — ITAM 2026
================================================================
  Lee el CSV de datos y el CSV de etiquetas generados por
  kmeans_serial o kmeans_parallel y produce gráficas de clusters.

  Uso:
    python visualize_clusters.py --data datos.csv --labels labels_serial.csv
    python visualize_clusters.py --data datos.csv --labels labels_parallel.csv --out clusters.png

  Opciones:
    --data     CSV de entrada (mismo que se le dio al binario)
    --labels   CSV de etiquetas generado por el binario
    --out      Nombre del archivo de salida (default: clusters.png)
    --title    Título de la gráfica (opcional)
    --centroids  CSV de centroides (opcional, para marcarlos)
================================================================
"""

import argparse
import os
import sys

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
except ImportError:
    print("[ERROR] matplotlib no está instalado. Corre: pip install matplotlib")
    sys.exit(1)


# ── Paleta de colores distinguibles ──────────────────────────────────────────
COLORS = [
    "#E63946",  # rojo
    "#457B9D",  # azul acero
    "#2A9D8F",  # verde azulado
    "#E9C46A",  # amarillo
    "#8338EC",  # violeta
    "#F4A261",  # naranja
    "#06D6A0",  # menta
    "#073B4C",  # azul marino
    "#FF6B6B",  # coral
    "#48CAE4",  # cyan
    "#A8DADC",  # celeste
    "#C77DFF",  # lavanda
]


def load_data(path):
    """Carga el CSV de datos. Detecta si es 2D o 3D por el número de columnas."""
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    dims = data.shape[1]
    print(f"[IO] Datos: {data.shape[0]:,} puntos, {dims}D — {path}")
    return data, dims


def load_labels(path):
    """Carga el CSV de etiquetas (un entero por línea)."""
    labels = np.loadtxt(path, dtype=int)
    k = len(np.unique(labels))
    print(f"[IO] Etiquetas: {len(labels):,} puntos, {k} clusters — {path}")
    return labels, k


def load_centroids(path, dims):
    """Carga el CSV de centroides si existe."""
    if not path or not os.path.isfile(path):
        return None
    c = np.loadtxt(path, delimiter=",")
    if c.ndim == 1:
        c = c.reshape(1, -1)
    # Si los centroides tienen más columnas que datos, truncar
    c = c[:, :dims]
    print(f"[IO] Centroides: {c.shape[0]} centroides — {path}")
    return c


# ── Submuestreo para datasets grandes ────────────────────────────────────────
def subsample(data, labels, max_points=50_000):
    """
    Si el dataset tiene más de max_points, submuestrea aleatoriamente.
    Matplotlib se vuelve lento con más de ~50k puntos en un scatter.
    """
    n = len(data)
    if n <= max_points:
        return data, labels
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=max_points, replace=False)
    print(f"[Info] Dataset grande ({n:,} puntos) — mostrando {max_points:,} aleatorios")
    return data[idx], labels[idx]


# ── Gráfica 2D ───────────────────────────────────────────────────────────────
def plot_2d(data, labels, k, centroids, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 7))

    for c in range(k):
        mask = labels == c
        color = COLORS[c % len(COLORS)]
        ax.scatter(
            data[mask, 0],
            data[mask, 1],
            s=4,
            alpha=0.25,
            color=color,
            label=f"Cluster {c}  ({mask.sum():,})",
            linewidths=0,
        )

    # Centroides
    if centroids is not None:
        for c in range(min(k, len(centroids))):
            color = COLORS[c % len(COLORS)]
            ax.scatter(
                centroids[c, 0],
                centroids[c, 1],
                s=250,
                marker="P",
                color=color,
                edgecolors="white",
                linewidths=1.8,
                zorder=5,
            )

    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.legend(
        fontsize=8,
        loc="upper right",
        markerscale=3,
        framealpha=0.85,
        ncol=2 if k > 6 else 1,
    )
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {out_path}")


# ── Gráfica 3D ───────────────────────────────────────────────────────────────
def plot_3d(data, labels, k, centroids, title, out_path):
    """
    Genera una cuadrícula 2×2 con cuatro vistas del mismo espacio 3D:
    perspectiva general, proyección XY, proyección XZ y proyección YZ.
    Así se puede apreciar la estructura sin depender de interactividad.
    """
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    views = [
        (221, "Vista 3D", 30, 45, True),
        (222, "Proyeccion XY", 90, -90, False),
        (223, "Proyeccion XZ", 0, -90, False),
        (224, "Proyeccion YZ", 0, 0, False),
    ]

    for pos, subtitle, elev, azim, show_z in views:
        ax = fig.add_subplot(pos, projection="3d")

        for c in range(k):
            mask = labels == c
            color = COLORS[c % len(COLORS)]
            ax.scatter(
                data[mask, 0],
                data[mask, 1],
                data[mask, 2],
                s=2,
                alpha=0.2,
                color=color,
                linewidths=0,
                label=f"C{c}" if pos == 221 else None,
            )

        # Centroides solo en la vista 3D
        if centroids is not None and pos == 221:
            for c in range(min(k, len(centroids))):
                ax.scatter(
                    [centroids[c, 0]],
                    [centroids[c, 1]],
                    [centroids[c, 2]],
                    s=200,
                    marker="P",
                    color=COLORS[c % len(COLORS)],
                    edgecolors="white",
                    linewidths=1.5,
                    zorder=5,
                )

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("X", fontsize=9, labelpad=4)
        ax.set_ylabel("Y", fontsize=9, labelpad=4)
        if show_z:
            ax.set_zlabel("Z", fontsize=9, labelpad=4)
        ax.set_title(subtitle, fontsize=11, pad=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.2)

        if pos == 221:
            ax.legend(
                fontsize=7,
                loc="upper right",
                markerscale=3,
                framealpha=0.8,
                ncol=2 if k > 6 else 1,
            )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {out_path}")


# ── Gráfica adicional: distribución de tamaños de cluster ────────────────────
def plot_cluster_sizes(labels, k, title, out_path):
    sizes = [int((labels == c).sum()) for c in range(k)]
    colors = [COLORS[c % len(COLORS)] for c in range(k)]

    fig, ax = plt.subplots(figsize=(max(6, k * 0.9), 5))
    bars = ax.bar(
        [f"C{c}" for c in range(k)],
        sizes,
        color=colors,
        edgecolor="white",
        linewidth=1.2,
    )
    ax.bar_label(bars, fmt="{:,.0f}", padding=4, fontsize=9)
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Puntos asignados", fontsize=12)
    ax.set_title(f"Distribucion de tamanios — {title}", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_facecolor("#FAFAFA")
    total = sum(sizes)
    ax.text(
        0.98,
        0.97,
        f"Total: {total:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#555555",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Visualiza los clusters generados por kmeans_serial / kmeans_parallel"
    )
    p.add_argument("--data", required=True, help="CSV de datos de entrada")
    p.add_argument("--labels", required=True, help="CSV de etiquetas (labels_*.csv)")
    p.add_argument("--centroids", default=None, help="CSV de centroides (opcional)")
    p.add_argument(
        "--out", default=None, help="Archivo de salida (default: clusters.png)"
    )
    p.add_argument("--title", default=None, help="Titulo personalizado")
    p.add_argument(
        "--max_pts",
        type=int,
        default=50_000,
        help="Max puntos a graficar (default: 50000)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Cargar datos
    data, dims = load_data(args.data)
    labels, k = load_labels(args.labels)

    if len(data) != len(labels):
        print(f"[ERROR] data tiene {len(data)} filas pero labels tiene {len(labels)}")
        sys.exit(1)

    centroids = load_centroids(args.centroids, dims)

    # Nombre base para los archivos de salida
    base = args.out if args.out else "clusters"
    base_noext = os.path.splitext(base)[0]

    # Título automático si no se especifica
    data_name = os.path.splitext(os.path.basename(args.data))[0]
    lbl_name = os.path.splitext(os.path.basename(args.labels))[0]
    title = args.title or f"K-Means — {data_name}  |  k={k}  |  {lbl_name}"

    print(f"\n[Config] dims={dims}  k={k}  n={len(data):,}")
    print(f"[Config] titulo: {title}\n")

    # Submuestrear si es necesario
    data_plot, labels_plot = subsample(data, labels, max_points=args.max_pts)

    # Gráfica principal de clusters
    cluster_out = f"{base_noext}.png"
    if dims == 2:
        plot_2d(data_plot, labels_plot, k, centroids, title, cluster_out)
    else:
        plot_3d(data_plot, labels_plot, k, centroids, title, cluster_out)

    # Gráfica de distribución de tamaños
    sizes_out = f"{base_noext}_sizes.png"
    plot_cluster_sizes(labels, k, title, sizes_out)

    # Resumen en consola
    print(f"\n{'='*50}")
    print(f"  Resumen de clusters")
    print(f"{'='*50}")
    total = len(labels)
    for c in range(k):
        n_c = int((labels == c).sum())
        pct = 100 * n_c / total
        bar = "█" * int(pct / 2)
        print(f"  Cluster {c:>2}: {n_c:>8,}  ({pct:5.1f}%)  {bar}")
    print(f"{'='*50}")
    print(f"  Total: {total:,} puntos")


if __name__ == "__main__":
    main()
