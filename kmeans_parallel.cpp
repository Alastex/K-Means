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
// ─────────────────────────────────────────────
using Point    = vector<double>;
using Dataset  = vector<Point>;
using Centroid = vector<double>;

// ─────────────────────────────────────────────
//  I/O
// ─────────────────────────────────────────────
Dataset load_csv(const string& path) {
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "[ERROR] No se puede abrir: " << path << "\n";
        exit(1);
    }
    Dataset data;
    string line;
    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string token;
        Point p;
        while (getline(ss, token, ','))
            p.push_back(stod(token));
        if (!p.empty()) data.push_back(std::move(p));
    }
    cout << "[IO] " << data.size() << " puntos, dim=" << data[0].size() << "\n";
    return data;
}

void save_labels(const string& path, const vector<int>& labels) {
    ofstream f(path);
    for (int l : labels) f << l << "\n";
}

void save_centroids(const string& path, const vector<Centroid>& centroids) {
    ofstream f(path);
    f << fixed << setprecision(6);
    for (const auto& c : centroids) {
        for (size_t d = 0; d < c.size(); ++d) {
            f << c[d];
            if (d + 1 < c.size()) f << ",";
        }
        f << "\n";
    }
}

// ─────────────────────────────────────────────
//  Distancia euclidiana al cuadrado (inline)
// ─────────────────────────────────────────────
inline double sq_dist(const Point& a, const Centroid& b) {
    double s = 0.0;
    for (size_t d = 0; d < a.size(); ++d) {
        double diff = a[d] - b[d];
        s += diff * diff;
    }
    return s;
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
void update_centroids_parallel(const Dataset& data,
                                const vector<int>& labels,
                                vector<Centroid>& centroids) {
    int    k        = static_cast<int>(centroids.size());
    size_t dim      = data[0].size();
    size_t n        = data.size();
    int    nthreads = omp_get_max_threads();

    // Pre-alocamos arreglos indexados por hilo:
    //   all_sums[t][c][d]  = suma parcial del hilo t para cluster c, dim d
    //   all_counts[t][c]   = cuántos puntos asignó el hilo t al cluster c
    // Sin sección crítica en el loop caliente — cada hilo escribe en su propia fila.
    vector<vector<Centroid>> all_sums(nthreads, vector<Centroid>(k, Centroid(dim, 0.0)));
    vector<vector<int>>      all_counts(nthreads, vector<int>(k, 0));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        // Cada hilo acumula en su propia partición — sin condición de carrera
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            int c = labels[i];
            for (size_t d = 0; d < dim; ++d)
                all_sums[tid][c][d] += data[i][d];
            ++all_counts[tid][c];
        }
    }   // fin de región paralela — barrera implícita

    // Reducción serial sobre los nthreads arreglos: O(nthreads * k * dim)
    // Esta parte es O(hilos × k × dim) ≪ O(n × k) — costo despreciable
    vector<Centroid> global_sums(k, Centroid(dim, 0.0));
    vector<int>      global_counts(k, 0);

    for (int t = 0; t < nthreads; ++t) {
        for (int c = 0; c < k; ++c) {
            for (size_t d = 0; d < dim; ++d)
                global_sums[c][d] += all_sums[t][c][d];
            global_counts[c] += all_counts[t][c];
        }
    }

    // Media final por cluster
    for (int c = 0; c < k; ++c) {
        if (global_counts[c] == 0) continue;
        for (size_t d = 0; d < dim; ++d)
            centroids[c][d] = global_sums[c][d] / global_counts[c];
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

    auto t0 = chrono::high_resolution_clock::now();

    int iter = 0;
    for (iter = 1; iter <= max_iter; ++iter) {
        int changes = assign_clusters_parallel(data, centroids, labels);
        update_centroids_parallel(data, labels, centroids);

        if (verbose && (iter == 1 || iter % 10 == 0))
            cout << "  iter=" << setw(3) << iter
                 << "  cambios=" << setw(7) << changes << "\n";

        if (changes == 0) {
            if (verbose)
                cout << "  [Convergió en iter " << iter << "]\n";
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

    Dataset data = load_csv(csv);
    kmeans_parallel(data, k, num_threads, max_iter, seed, true, true);

    return 0;
}
