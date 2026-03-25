/**
 * ============================================================
 *  K-Means — Versión Paralela con OpenMP (Proyecto Apertura)
 *  Cómputo Paralelo y en la Nube — ITAM 2026
 * ============================================================
 *
 *  Esta versión toma el algoritmo serial y paraleliza exactamente
 *  las dos operaciones que dominan el tiempo de ejecución:
 *  la asignación de clusters (E-step) y la actualización de
 *  centroides (M-step).
 *
 *  Respecto a la versión serial, hay tres decisiones de diseño
 *  importantes que se explican a detalle abajo:
 *
 *  1. struct Point en lugar de vector<double>
 *     El layout de memoria importa para el paralelismo: con un
 *     struct fijo el procesador puede hacer prefetch y el compilador
 *     puede vectorizar. Con vector<double>, cada punto es un heap
 *     allocation separado que rompe la localidad de caché.
 *
 *  2. SoA (Structure of Arrays) para los acumuladores del M-step
 *     En lugar de all_sums[tid][c][d] con triple indirección, usamos
 *     lsX[tid][c], lsY[tid][c], lsZ[tid][c] — arreglos planos que
 *     el compilador puede vectorizar con AVX2/SSE4.
 *
 *  3. Fusión E-step + M-step con nowait
 *     Elimina una barrera de sincronización por iteración del loop
 *     de Lloyd. Con datos que convergen en pocas iteraciones, cada
 *     barrera eliminada tiene impacto visible.
 *
 *  Build:
 *    g++ -O3 -std=c++17 -fopenmp -o kmeans_parallel kmeans_parallel.cpp
 *    (Nota: -O3 en lugar de -O2 activa vectorización SIMD más agresiva)
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
#include <omp.h>        // API de OpenMP: omp_get_thread_num, omp_set_num_threads, etc.
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// ─────────────────────────────────────────────
//  struct Point — layout fijo en lugar de vector<double>
//
//  Esta es la diferencia de representación más importante
//  respecto a la versión serial.
//
//  Con vector<double>, cada punto es:
//    [ puntero(8B) | tamaño(8B) | capacidad(8B) ] → datos en el heap
//  Con un millón de puntos tenemos un millón de punteros esparcidos
//  por el heap. El procesador no puede prefetchear la memoria porque
//  no sabe dónde está el siguiente punto.
//
//  Con struct Point { double x, y, z }, los datos quedan así en RAM:
//    [x0,y0,z0, x1,y1,z1, x2,y2,z2, ...]
//  Un bloque contiguo de 24 bytes por punto. El hardware prefetcher
//  del CPU detecta el patrón secuencial y carga las líneas de caché
//  antes de que el código las necesite.
//
//  La limitación: este struct solo funciona hasta 3D. Si necesitáramos
//  dimensionalidad arbitraria, habría que volver a vector<double> o
//  usar un arreglo fijo con una constante de compilación.
// ─────────────────────────────────────────────
struct Point {
    double x = 0, y = 0, z = 0;
};
using Dataset  = vector<Point>;
using Centroid = Point;  // mismo tipo — son intercambiables semánticamente

// ─────────────────────────────────────────────
//  load_csv
//
//  Igual que en el serial pero adaptado a struct Point.
//  dim se pasa por referencia para que el caller pueda saber
//  si el CSV era 2D o 3D sin leerlo dos veces.
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
//  sq_dist — versión optimizada para struct Point
//
//  Al tener campos con nombre (x, y, z) en lugar de un loop genérico,
//  el compilador genera exactamente 3 restas, 3 multiplicaciones y
//  2 sumas — sin overhead de loop ni contador. Con -O3 esto puede
//  compilar a una sola instrucción SIMD que procesa las 3 componentes
//  en paralelo dentro del mismo core.
//
//  Comparación con el serial:
//    Serial:   for (d=0..dim) diff = a[d]-b[d]; s += diff*diff;
//    Paralelo: dx=a.x-b.x; dy=...; dz=...; return dx*dx+dy*dy+dz*dz;
//  Mismo resultado, pero sin el overhead del loop y con mejor
//  oportunidad de vectorización.
// ─────────────────────────────────────────────
inline double sq_dist(const Point& a, const Centroid& b) {
    double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

// ─────────────────────────────────────────────
//  kmeanspp_init — igual que en el serial, corre una sola vez
//
//  Intencionalmente no paralelizado: corre una vez antes del loop
//  de Lloyd y su costo (O(k*n)) es despreciable frente al costo
//  total del algoritmo. Añadir pragmas aquí complicaría el código
//  sin ganancia medible.
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
//  kmeans_parallel — driver principal con E-step y M-step fusionados
//
//  La decisión de diseño más importante aquí es fusionar ambos pasos
//  en una sola región paralela con nowait. Explicación:
//
//  Sin fusión (lo que hacíamos antes):
//    Iteración:
//      [E-step paralelo] → BARRERA → [M-step paralelo] → BARRERA → check
//
//  Con fusión y nowait:
//    Iteración:
//      [E-step + M-step en una región] → BARRERA → check
//
//  Nos ahorramos una barrera por iteración. Con datos que convergen
//  en 10 iteraciones, son 10 barreras menos — cada una cuesta entre
//  5 y 50 microsegundos dependiendo del hardware y el número de hilos.
//
//  La fusión es segura porque schedule(static) garantiza que el hilo T
//  procesa exactamente el mismo rango de índices [a, b) en AMBOS loops.
//  Cuando el hilo T empieza el M-step para el índice i, ya escribió
//  labels[i] en el E-step — nadie más toca ese índice.
// ─────────────────────────────────────────────
double kmeans_parallel(const Dataset& data, int k, int num_threads,
                       int max_iter = 300, unsigned seed = 42,
                       bool verbose = true,
                       bool save = true) {
    size_t n = data.size();

    // omp_set_num_threads fija el número de hilos que usarán TODAS las
    // regiones paralelas de este proceso a partir de este punto.
    // Alternativa: OMP_NUM_THREADS env var, pero el parámetro explícito
    // es más predecible en benchmarks.
    omp_set_num_threads(num_threads);

    mt19937 rng(seed);
    auto centroids = kmeanspp_init(data, k, rng);
    vector<int> labels(n, -1);

    int nthreads = num_threads;

    // ── Pre-alocación de buffers SoA fuera del loop ─────────────────────
    //
    // Por qué alocar aquí y no dentro de update_centroids_parallel:
    // Si alocaramos dentro del loop de Lloyd, haríamos malloc/free en
    // cada iteración. Con n=1M y 10 iteraciones, son 10 mallocs de
    // nthreads * k doubles cada uno. No son costosos individualmente,
    // pero suman y fragmentan el heap.
    //
    // La alternativa clásica sería pasar los buffers como parámetros,
    // pero como fusionamos E y M en una sola región paralela, los
    // buffers viven en el scope del driver directamente.
    //
    // SoA: lsX[tid][c] en lugar de sums[tid][c].x
    // Ventaja: lsX[0..nthreads-1][c] son nthreads doubles consecutivos
    // en memoria para el cluster c. El compilador puede vectorizar la
    // reducción final con instrucciones SIMD.
    vector<vector<double>> lsX(nthreads, vector<double>(k, 0.0));
    vector<vector<double>> lsY(nthreads, vector<double>(k, 0.0));
    vector<vector<double>> lsZ(nthreads, vector<double>(k, 0.0));
    vector<vector<int>>    lsCnt(nthreads, vector<int>(k, 0));

    auto t0 = chrono::high_resolution_clock::now();

    int iter = 0;
    for (iter = 1; iter <= max_iter; ++iter) {

        int changes = 0;

        // ── Región paralela: E-step + M-step fusionados ─────────────────
        //
        // #pragma omp parallel abre el team de hilos y define el scope
        // compartido. Todo lo declarado DENTRO de este bloque es
        // thread-private (tid, best_d, best_c, etc.).
        // Todo lo declarado FUERA (data, centroids, labels, lsX...) es
        // compartido por todos los hilos — pero cada uno escribe en
        // posiciones distintas, así que no hay races.
        //
        // reduction(+:changes): OpenMP crea una copia privada de changes
        // en cada hilo (inicializada a 0), y al cerrar el parallel suma
        // todas las copias en la variable original. Es equivalente a
        // atomic pero sin serializar el loop interno.
        #pragma omp parallel reduction(+:changes)
        {
            int tid = omp_get_thread_num();

            // Reset de los buffers locales. fill() es O(k) — con k=8
            // son 8 asignaciones, completamente despreciable.
            // Nota: hacemos esto DENTRO de la región paralela para que
            // cada hilo limpie su propia fila, en paralelo con los demás.
            fill(lsX[tid].begin(),   lsX[tid].end(),   0.0);
            fill(lsY[tid].begin(),   lsY[tid].end(),   0.0);
            fill(lsZ[tid].begin(),   lsZ[tid].end(),   0.0);
            fill(lsCnt[tid].begin(), lsCnt[tid].end(), 0);

            // ── E-step paralelo con nowait ──────────────────────────────
            //
            // schedule(static): divide [0, n) en bloques de n/nthreads y
            // asigna cada bloque a un hilo en tiempo de compilación.
            // Es la política correcta aquí porque el trabajo por iteración
            // es homogéneo: todas las distancias cuestan lo mismo.
            // schedule(dynamic) añadiría overhead de sincronización sin
            // ningún beneficio de balance de carga.
            //
            // nowait: el hilo NO espera a que los demás terminen el E-step
            // antes de pasar al M-step. Es seguro porque:
            //   1. schedule(static) garantiza mismo rango en ambos loops
            //   2. El hilo T terminó de escribir labels[i] para su rango
            //      antes de leerlos en el M-step para ese mismo rango
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
                    ++changes;  // se suma en la reducción al cerrar el parallel
                }
            }

            // ── M-step paralelo con SoA ─────────────────────────────────
            //
            // Cada hilo acumula en su propia fila lsX[tid], lsY[tid], lsZ[tid].
            // No hay dos hilos escribiendo en la misma posición de memoria,
            // por lo que no se necesita atomic, critical, ni lock de ningún tipo.
            //
            // Si usáramos un arreglo global sums[c] sin protección, dos hilos
            // que tengan puntos del mismo cluster c harían:
            //   hilo 0: sums[c] += data[i0].x   ← lee y escribe
            //   hilo 1: sums[c] += data[i1].x   ← lee y escribe (race condition)
            //
            // Con lsX[tid][c] cada hilo escribe en su propia celda:
            //   hilo 0: lsX[0][c] += data[i0].x  ← solo el hilo 0 toca lsX[0]
            //   hilo 1: lsX[1][c] += data[i1].x  ← solo el hilo 1 toca lsX[1]
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                int c = labels[i];
                lsX[tid][c] += data[i].x;
                lsY[tid][c] += data[i].y;
                lsZ[tid][c] += data[i].z;
                ++lsCnt[tid][c];
            }

        } // ← barrera implícita: todos los hilos terminaron E+M antes de continuar

        // ── Reducción serial de los buffers SoA ─────────────────────────
        //
        // Costo: O(nthreads * k). Con 32 hilos y k=8 son 256 sumas.
        // Frente al O(n * k) del loop paralelo (8 millones de distancias
        // con n=1M y k=8), esto es 31,250x más pequeño — completamente
        // despreciable. No tiene sentido paralelizar esto.
        for (int c = 0; c < k; ++c) {
            double sx = 0, sy = 0, sz = 0;
            int    cnt = 0;
            for (int t = 0; t < nthreads; ++t) {
                sx  += lsX[t][c];
                sy  += lsY[t][c];
                sz  += lsZ[t][c];
                cnt += lsCnt[t][c];
            }
            // El guard evita división por cero si un cluster queda vacío
            if (cnt > 0) {
                centroids[c].x = sx / cnt;
                centroids[c].y = sy / cnt;
                centroids[c].z = sz / cnt;
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
        // El benchmark parsea "tiempo=X.XXXXs" de este output
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
//
//  Igual que en el serial: parsear, cargar, ejecutar.
//  La diferencia es num_threads como argumento obligatorio,
//  para que el benchmark pueda controlar el número de hilos
//  desde Python sin variables de entorno.
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