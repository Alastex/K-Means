/**
 * ============================================================
 *  K-Means — Versión Serial (Proyecto Apertura)
 *  Cómputo Paralelo y en la Nube — ITAM 2026
 * ============================================================
 *  Algoritmo:
 *    Paso 1. Inicializar k centroides (K-Means++)
 *    Paso 2. Asignar cada punto al centroide más cercano
 *    Paso 3. Recalcular centroides como promedio de sus puntos
 *    Repetir pasos 2-3 hasta que ningún punto cambie de cluster
 *
 *  Build:
 *    g++ -O2 -std=c++17 -o kmeans_serial kmeans_serial.cpp
 *
 *  Uso:
 *    ./kmeans_serial <csv_file> <k> [max_iter=300] [seed=42]
 *
 *  Salida:
 *    labels_serial.csv    — cluster asignado a cada punto
 *    centroids_serial.csv — posición final de los k centroides
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
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// ─────────────────────────────────────────────
//  Tipos
// ─────────────────────────────────────────────
using Point = vector<double>;
using Dataset = vector<Point>;
using Centroid = vector<double>;

// ─────────────────────────────────────────────
//  I/O
// ─────────────────────────────────────────────
Dataset load_csv(const string &path)
{
    // Abre archivo CSV de puntos (cada línea es un punto, valores separados por coma)
    ifstream file(path);

    if (!file.is_open())
    {
        cerr << "[ERROR] No se puede abrir: " << path << "\n";
        exit(1);
    }

    Dataset data;
    string line;
    while (getline(file, line))
    {
        if (line.empty())
            continue;

        // Parsear cada línea en valores numéricos
        stringstream ss(line);
        string token;
        Point p;
        while (getline(ss, token, ','))
            p.push_back(stod(token));

        // Ignorar líneas vacías, mover el punto al dataset
        if (!p.empty())
            data.push_back(std::move(p));
    }

    // Mostrar dimensión y número de puntos cargados
    cout << "[IO] " << data.size() << " puntos, dim=" << data[0].size() << "\n";
    return data;
}

void save_labels(const string &path, const vector<int> &labels)
{
    // Guardar asignaciones de cluster para cada punto
    ofstream f(path);
    for (int l : labels)
        f << l << "\n";
}

void save_centroids(const string &path, const vector<Centroid> &centroids)
{
    // Guardar posiciones finales de los centroides con 6 decimales
    ofstream f(path);
    f << fixed << setprecision(6);
    for (const auto &c : centroids)
    {
        for (size_t d = 0; d < c.size(); ++d)
        {
            f << c[d];
            if (d + 1 < c.size())
                f << ",";
        }
        f << "\n";
    }
}

// ─────────────────────────────────────────────
//  Distancia euclidiana al cuadrado (coste de cálculo menor que distancia completa)
// ─────────────────────────────────────────────
inline double sq_dist(const Point &a, const Centroid &b)
{
    double s = 0.0;
    for (size_t d = 0; d < a.size(); ++d)
    {
        double diff = a[d] - b[d];
        s += diff * diff;
    }
    return s;
}

// ─────────────────────────────────────────────
//  K-Means++ — inicialización inteligente
//  Probabilidad de elegir punto i ∝ D²(i, centroide más cercano)
// ─────────────────────────────────────────────
vector<Centroid> kmeanspp_init(const Dataset &data, int k, mt19937 &rng)
{
    // Inicializacion K-Means++:
    // 1) Escoge 1 centroide al azar
    // 2) Para cada punto, calcula distancia al centroide más cercano
    // 3) Selecciona siguiente centroide con probabilidad proporcional a dist^2
    // Repetir hasta tener k centroides.

    size_t n = data.size();
    vector<Centroid> centroids;
    centroids.reserve(k);

    uniform_int_distribution<size_t> unif(0, n - 1);
    centroids.push_back(data[unif(rng)]);  // primer centroide aleatorio

    vector<double> dist2(n, numeric_limits<double>::infinity());

    for (int c = 1; c < k; ++c)
    {
        const Centroid &last = centroids.back();
        for (size_t i = 0; i < n; ++i)
        {
            double d = sq_dist(data[i], last);
            if (d < dist2[i])
                dist2[i] = d;  // dist^2 al centroide más cercano hasta ahora
        }

        // El siguiente centroide se elige con distribution ponderada por dist2
        discrete_distribution<size_t> weighted(dist2.begin(), dist2.end());
        centroids.push_back(data[weighted(rng)]);
    }
    return centroids;
}

// ─────────────────────────────────────────────
//  Paso 2 — Asignación (E-step)  [SERIAL]
//  Retorna número de puntos que cambiaron de cluster
// ─────────────────────────────────────────────
int assign_clusters(const Dataset &data,
                    const vector<Centroid> &centroids,
                    vector<int> &labels)
{
    // Paso E (expectation): asigna cada punto al centroide más cercano
    // Devuelve el número de cambios de cluster (criterio de convergencia)
    int changes = 0;
    int k = static_cast<int>(centroids.size());

    for (size_t i = 0; i < data.size(); ++i)
    {
        double best_d = numeric_limits<double>::infinity();
        int best_c = 0;
        for (int c = 0; c < k; ++c)
        {
            double d = sq_dist(data[i], centroids[c]);
            if (d < best_d)
            {
                best_d = d;
                best_c = c;
            }
        }

        // Si la etiqueta cambia, incrementar cambios
        if (labels[i] != best_c)
        {
            labels[i] = best_c;
            ++changes;
        }
    }
    return changes;
}

// ─────────────────────────────────────────────
//  Paso 3 — Actualización de centroides (M-step)  [SERIAL]
//  Centroide_c = promedio de posiciones de sus puntos
// ─────────────────────────────────────────────
void update_centroids(const Dataset &data,
                      const vector<int> &labels,
                      vector<Centroid> &centroids)
{
    // Paso M (maximization): recalcula cada centroide como promedio
    // de todos los puntos asignados a ese cluster.
    int k = static_cast<int>(centroids.size());
    size_t dim = data[0].size();

    vector<Centroid> sums(k, Centroid(dim, 0.0));  // sumas por cluster
    vector<int> counts(k, 0);                       // tamaños de cluster

    for (size_t i = 0; i < data.size(); ++i)
    {
        int c = labels[i];
        for (size_t d = 0; d < dim; ++d)
            sums[c][d] += data[i][d];  // acumular coordenadas
        ++counts[c];
    }

    for (int c = 0; c < k; ++c)
    {
        if (counts[c] == 0)
            continue;  // evitar división por cero (cluster vacío)
        for (size_t d = 0; d < dim; ++d)
            centroids[c][d] = sums[c][d] / counts[c];
    }
}

// ─────────────────────────────────────────────
//  K-Means serial — driver principal
//  Retorna tiempo de ejecución en segundos
// ─────────────────────────────────────────────
double kmeans_serial(const Dataset &data, int k,
                     int max_iter = 300, unsigned seed = 42,
                     bool verbose = true,
                     bool save = true)
{
    // Driver principal del algoritmo serial.
    // - Inicializa centroides (K-Means++)
    // - Repite asignar->actualizar hasta convergencia o max_iter
    size_t n = data.size();
    mt19937 rng(seed);

    auto centroids = kmeanspp_init(data, k, rng);  // inicializacion inteligente
    vector<int> labels(n, -1);                     // etiquetas de cluster

    auto t0 = chrono::high_resolution_clock::now();

    int iter = 0;
    for (iter = 1; iter <= max_iter; ++iter)
    {
        int changes = assign_clusters(data, centroids, labels);  // E-step
        update_centroids(data, labels, centroids);               // M-step

        if (verbose && (iter == 1 || iter % 10 == 0))
            cout << "  iter=" << setw(3) << iter
                 << "  cambios=" << setw(7) << changes << "\n";

        // Criterio de parada: no cambió ningún punto de cluster
        if (changes == 0)
        {
            if (verbose)
                cout << "  [Convergió en iter " << iter << "]\n";
            break;
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(t1 - t0).count();

    if (verbose)
    {
        cout << "[Serial] iters=" << iter
             << "  tiempo=" << fixed << setprecision(4) << elapsed << "s\n";
        vector<int> sizes(k, 0);
        for (int l : labels)
            ++sizes[l];
        for (int c = 0; c < k; ++c)
            cout << "  Cluster " << c << ": " << sizes[c] << " puntos\n";
    }

    if (save)
    {
        save_labels("labels_serial.csv", labels);
        save_centroids("centroids_serial.csv", centroids);
        if (verbose)
            cout << "[IO] labels_serial.csv  centroids_serial.csv\n";
    }

    return elapsed;
}

// ─────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────
int main(int argc, char *argv[])
{
    // Validación de argumentos de línea de comandos
    if (argc < 3)
    {
        cerr << "Uso: " << argv[0]
             << " <csv_file> <k> [max_iter=300] [seed=42]\n";
        return 1;
    }

    string csv = argv[1];
    int k = stoi(argv[2]);
    int max_iter = (argc > 3) ? stoi(argv[3]) : 300;
    unsigned seed = (argc > 4) ? stoul(argv[4]) : 42;

    cout << "══════════════════════════════════════════\n";
    cout << " K-Means Serial  k=" << k
         << "  max_iter=" << max_iter << "\n";
    cout << "══════════════════════════════════════════\n";

    Dataset data = load_csv(csv);

    // Ejecuta el algoritmo serial y guarda resultados
    kmeans_serial(data, k, max_iter, seed, true, true);

    return 0;
}
