/**
 * ============================================================
 *  K-Means — Versión Paralela con OpenMP (Proyecto Apertura)
 *  Cómputo Paralelo y en la Nube — ITAM 2026
 * ============================================================
 *  Estrategia de paralelización:
 *
 *  Paso 2 — E-step (Asignación):
 *    El loop principal sobre los n puntos es embarazosamente
 *    paralelo: cada punto calcula su distancia a los k centroides
 *    de forma completamente independiente.
 *    → #pragma omp parallel for reduction(+:changes)
 *      con schedule(static) para distribución uniforme.
 *
 *  Paso 3 — M-step (Actualización de centroides):
 *    Cada hilo acumula sumas parciales en arreglos locales
 *    (thread-private), evitando condiciones de carrera sin
 *    usar atomic/critical en el loop caliente.
 *    Al final, una sección crítica combina los resultados.
 *    → Pattern: partial-sum reduction manual sobre vectores.
 *
 *  Build:
 *    g++ -O2 -std=c++17 -fopenmp -o kmeans_parallel kmeans_parallel.cpp
 *
 *  Uso:
 *    ./kmeans_parallel <csv_file> <k> <num_threads> [max_iter=300] [seed=42]
 *
 *  Salida:
 *    labels_parallel.csv    — cluster asignado a cada punto
 *    centroids_parallel.csv — posición final de los k centroides
 * ============================================================
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// ─────────────────────────────────────────────
//  Tipos
//  FIX 3: struct fijo en lugar de vector<double>
//  Con vector<double>, cada uno de los n puntos es un heap
//  allocation separado. Con struct Point los datos quedan en
//  un bloque contiguo de memoria — el CPU puede hacer prefetch
//  y el compilador puede vectorizar (SIMD) los accesos.
// ─────────────────────────────────────────────
struct Point {
    double x = 0, y = 0, z = 0;
};
using Dataset  = vector<Point>;
using Centroid = Point;

// ─────────────────────────────────────────────
//  I/O
// ─────────────────────────────────────────────
Dataset load_csv(const string& path, int& dim) {
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "[ERROR] No se puede abrir: " << path << "\n";
        exit(1);
    }
    Dataset data;
    string line;
    dim = 0;
    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string tok;
        vector<double> vals;
        while (getline(ss, tok, ','))
            vals.push_back(stod(tok));
        if (dim == 0) dim = (int)vals.size();
        Point p;
        p.x = vals[0];
        p.y = (vals.size() > 1) ? vals[1] : 0.0;
        p.z = (vals.size() > 2) ? vals[2] : 0.0;
        data.push_back(p);
    }
    cout << "[IO] " << data.size() << " puntos, dim=" << dim << "\n";
    return data;
}

void save_labels(const string& path, const vector<int>& labels) {
    ofstream f(path);
    for (int l : labels) f << l << "\n";
}

void save_centroids(const string& path, const vector<Centroid>& centroids) {
    ofstream f(path);
    f << fixed << setprecision(6);
    for (const auto& c : centroids)
        f << c.x << "," << c.y << "," << c.z << "\n";
}

// ─────────────────────────────────────────────
//  Distancia euclidiana al cuadrado (inline)
// ─────────────────────────────────────────────
// FIX 3: sq_dist sobre struct fijo — sin loop, sin indirección
inline double sq_dist(const Point& a, const Centroid& b) {
    double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

// ─────────────────────────────────────────────
//  K-Means++ — inicialización (serial — se ejecuta una vez)
// ─────────────────────────────────────────────
vector<Centroid> kmeanspp_init(const Dataset& data, int k, mt19937& rng) {
    size_t n = data.size();
    vector<Centroid> centroids;
    centroids.reserve(k);

    uniform_int_distribution<size_t> unif(0, n - 1);
    centroids.push_back(data[unif(rng)]);

    vector<double> dist2(n, numeric_limits<double>::infinity());

    for (int c = 1; c < k; ++c) {
        const Centroid& last = centroids.back();
        for (size_t i = 0; i < n; ++i) {
            double d = sq_dist(data[i], last);
            if (d < dist2[i]) dist2[i] = d;
        }
        discrete_distribution<size_t> weighted(dist2.begin(), dist2.end());
        centroids.push_back(data[weighted(rng)]);
    }
    return centroids;
}

// ─────────────────────────────────────────────
//  Paso 2 — Asignación paralela (E-step)
//
//  Esta función hace exactamente lo mismo que assign_clusters() en el
//  código serial, pero paraleliza el loop que itera sobre cada punto.
//  Comparación con serial:
//   - Serial: loop for (i=0..n) único y secuencial.
//   - Paralelo: #pragma omp parallel for, cada hilo procesa un subconjunto.
//  - `labels` es shared; cada hilo actualiza su índice i único.
//  - `changes` se acumula con reduction(+:changes) (evita critical).
//
//  Nota clave: `sq_dist` sigue siendo la misma función serial. No hay
//  atomics en el inner loop: la independencia por punto lo permite.
// ─────────────────────────────────────────────
int assign_clusters_parallel(const Dataset& data,
                              const vector<Centroid>& centroids,
                              vector<int>& labels) {
    int    changes = 0;
    int    k       = static_cast<int>(centroids.size());
    size_t n       = data.size();

    #pragma omp parallel for schedule(static) reduction(+:changes)
    for (size_t i = 0; i < n; ++i) {
        double best_d = numeric_limits<double>::infinity();
        int    best_c = 0;
        for (int c = 0; c < k; ++c) {
            double d = sq_dist(data[i], centroids[c]);
            if (d < best_d) { best_d = d; best_c = c; }
        }
        if (labels[i] != best_c) {
            labels[i] = best_c;
            ++changes;
        }
    }
    return changes;
}

// ─────────────────────────────────────────────
//  Paso 3 — Actualización paralela de centroides (M-step)
//
//  Comparado con la versión serial (update_centroids):
//   - Serial: suma directa global y conteo global en un solo loop.
//   - Paralelo: cada hilo acumula en estructuras locales (all_sums/all_counts)
//     para evitar condiciones de carrera sobre los datos compartidos.
//   - Luego se reduce (serialmente) a global_sums/global_counts.
//   - Evita bloqueo en el loop caliente, y el overhead de reducción es menor.
//
//  Paralelismo:
//    #pragma omp parallel + #pragma omp for sobre i
//    cada hilo trabaja con su índice tid y datos thread-private.
//
//  Elemento clave: no existe concurrencia en write sobre centroids/data durante
//  el loop pesado; solo en la fase de reducción (menor costo).
// ─────────────────────────────────────────────
// FIX 3: SoA (Structure of Arrays) — accesos planos de dos niveles
// en lugar de all_sums[tid][c][d] con triple indirección.
// Permite al compilador vectorizar con AVX2/SSE4 en el Ryzen 7 6800HS.
//
// Signatura extendida: recibe los buffers pre-alocados para no
// hacer malloc/free en cada iteración del loop de Lloyd.
void update_centroids_parallel(const Dataset& data,
                                const vector<int>& labels,
                                vector<Centroid>& centroids,
                                vector<vector<double>>& lsX,
                                vector<vector<double>>& lsY,
                                vector<vector<double>>& lsZ,
                                vector<vector<int>>&    lsCnt) {
    int    k        = static_cast<int>(centroids.size());
    size_t n        = data.size();
    int    nthreads = omp_get_max_threads();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        // Reset local sin re-alocar (fill es O(k) — despreciable)
        fill(lsX[tid].begin(), lsX[tid].end(), 0.0);
        fill(lsY[tid].begin(), lsY[tid].end(), 0.0);
        fill(lsZ[tid].begin(), lsZ[tid].end(), 0.0);
        fill(lsCnt[tid].begin(), lsCnt[tid].end(), 0);

        // M-step: acumular en SoA local — sin lock, sin false sharing
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            int c = labels[i];
            lsX[tid][c] += data[i].x;
            lsY[tid][c] += data[i].y;
            lsZ[tid][c] += data[i].z;
            ++lsCnt[tid][c];
        }
    }

    // Reducción serial: O(nthreads * k) — con k=8 y 32 hilos son 256 ops
    for (int c = 0; c < k; ++c) {
        double sx = 0, sy = 0, sz = 0;
        int    cnt = 0;
        for (int t = 0; t < nthreads; ++t) {
            sx  += lsX[t][c];
            sy  += lsY[t][c];
            sz  += lsZ[t][c];
            cnt += lsCnt[t][c];
        }
        if (cnt > 0) {
            centroids[c].x = sx / cnt;
            centroids[c].y = sy / cnt;
            centroids[c].z = sz / cnt;
        }
    }
}

// ─────────────────────────────────────────────
//  K-Means paralelo — driver principal
//  Retorna tiempo de ejecución en segundos
// ─────────────────────────────────────────────
double kmeans_parallel(const Dataset& data, int k, int num_threads,
                       int max_iter = 300, unsigned seed = 42,
                       bool verbose = true,
                       bool save = true) {
    size_t n = data.size();
    omp_set_num_threads(num_threads);

    mt19937 rng(seed);
    auto centroids = kmeanspp_init(data, k, rng);
    vector<int> labels(n, -1);

    int nthreads = num_threads;

    // FIX 3: Pre-alocar buffers SoA UNA sola vez fuera del loop.
    // Sin esto se hacen malloc/free en cada iteración.
    vector<vector<double>> lsX(nthreads, vector<double>(k, 0.0));
    vector<vector<double>> lsY(nthreads, vector<double>(k, 0.0));
    vector<vector<double>> lsZ(nthreads, vector<double>(k, 0.0));
    vector<vector<int>>    lsCnt(nthreads, vector<int>(k, 0));

    auto t0 = chrono::high_resolution_clock::now();

    int iter = 0;
    for (iter = 1; iter <= max_iter; ++iter) {
        // FIX 3: E-step y M-step en una sola región paralela con nowait.
        // nowait elimina la barrera entre el primer omp for y el segundo:
        // cada hilo empieza a acumular en cuanto termina su chunk de asignación.
        // Es seguro porque schedule(static) garantiza que ambos loops asignan
        // exactamente el mismo rango de índices al mismo hilo.
        int changes = 0;

        #pragma omp parallel reduction(+:changes)
        {
            int tid = omp_get_thread_num();

            fill(lsX[tid].begin(), lsX[tid].end(), 0.0);
            fill(lsY[tid].begin(), lsY[tid].end(), 0.0);
            fill(lsZ[tid].begin(), lsZ[tid].end(), 0.0);
            fill(lsCnt[tid].begin(), lsCnt[tid].end(), 0);

            // E-step: nowait — el hilo no espera a los demás
            #pragma omp for schedule(static) nowait
            for (size_t i = 0; i < n; ++i) {
                double best_d = numeric_limits<double>::infinity();
                int    best_c = 0;
                for (int c = 0; c < k; ++c) {
                    double d = sq_dist(data[i], centroids[c]);
                    if (d < best_d) { best_d = d; best_c = c; }
                }
                if (labels[i] != best_c) {
                    labels[i] = best_c;
                    ++changes;
                }
            }

            // M-step: mismo rango que E-step → labels[i] ya escrito
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                int c = labels[i];
                lsX[tid][c] += data[i].x;
                lsY[tid][c] += data[i].y;
                lsZ[tid][c] += data[i].z;
                ++lsCnt[tid][c];
            }
        }

        // Reducción serial de buffers SoA
        for (int c = 0; c < k; ++c) {
            double sx = 0, sy = 0, sz = 0; int cnt = 0;
            for (int t = 0; t < nthreads; ++t) {
                sx += lsX[t][c]; sy += lsY[t][c]; sz += lsZ[t][c];
                cnt += lsCnt[t][c];
            }
            if (cnt > 0) {
                centroids[c].x = sx/cnt;
                centroids[c].y = sy/cnt;
                centroids[c].z = sz/cnt;
            }
        }

        if (verbose && (iter == 1 || iter % 10 == 0))
            cout << "  iter=" << setw(3) << iter
                 << "  cambios=" << setw(7) << changes << "\n";

        if (changes == 0) {
            if (verbose)
                cout << "  [Convergio en iter " << iter << "]\n";
            break;
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(t1 - t0).count();

    if (verbose) {
        cout << "[Paralelo] hilos=" << num_threads
             << "  iters=" << iter
             << "  tiempo=" << fixed << setprecision(4) << elapsed << "s\n";
        vector<int> sizes(k, 0);
        for (int l : labels) ++sizes[l];
        for (int c = 0; c < k; ++c)
            cout << "  Cluster " << c << ": " << sizes[c] << " puntos\n";
    }

    if (save) {
        save_labels("labels_parallel.csv", labels);
        save_centroids("centroids_parallel.csv", centroids);
        if (verbose) cout << "[IO] labels_parallel.csv  centroids_parallel.csv\n";
    }

    return elapsed;
}

// ─────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Uso: " << argv[0]
             << " <csv_file> <k> <num_threads> [max_iter=300] [seed=42]\n";
        return 1;
    }

    string   csv         = argv[1];
    int      k           = stoi(argv[2]);
    int      num_threads = stoi(argv[3]);
    int      max_iter    = (argc > 4) ? stoi(argv[4]) : 300;
    unsigned seed        = (argc > 5) ? stoul(argv[5]) : 42;

    cout << "══════════════════════════════════════════════\n";
    cout << " K-Means Paralelo (OpenMP)  k=" << k
         << "  hilos=" << num_threads
         << "  max_iter=" << max_iter << "\n";
    cout << "══════════════════════════════════════════════\n";

    int dim = 0;
    Dataset data = load_csv(csv, dim);
    kmeans_parallel(data, k, num_threads, max_iter, seed, true, true);

    return 0;
}