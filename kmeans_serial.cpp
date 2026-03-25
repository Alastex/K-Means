/**
 * ============================================================
 *  K-Means — Versión Serial (Proyecto Apertura)
 *  Cómputo Paralelo y en la Nube — ITAM 2026
 * ============================================================
 *
 *  Implementación del algoritmo de Lloyd (K-Means) desde cero.
 *  Esta versión es intencionalmente serial y sirve como baseline
 *  para medir el speedup de la versión paralela con OpenMP.
 *
 *  Por qué esta estructura:
 *    El algoritmo tiene tres fases claramente separadas —
 *    inicialización, asignación y actualización — y las
 *    separamos en funciones distintas para que la versión
 *    paralela pueda reemplazar exactamente las que necesita
 *    sin tocar el resto del código.
 *
 *  Build:
 *    g++ -O2 -std=c++17 -o kmeans_serial kmeans_serial.cpp
 *
 *  Uso:
 *    ./kmeans_serial <csv_file> <k> [max_iter=300] [seed=42]
 *
 *  Salida:
 *    labels_serial.csv    — un entero por línea: el cluster de cada punto
 *    centroids_serial.csv — coordenadas finales de los k centroides
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
//  Aliases de tipo
//
//  Usamos aliases en lugar de structs para mantener el código
//  genérico: funciona con 2D, 3D o cualquier dimensión sin cambios.
//  Point y Centroid son el mismo tipo subyacente (vector<double>),
//  pero tenerlos como alias distintos hace que el código se lea
//  con la intención de quien lo escribió.
// ─────────────────────────────────────────────
using Point    = vector<double>;
using Dataset  = vector<Point>;
using Centroid = vector<double>;

// ─────────────────────────────────────────────
//  load_csv
//
//  Lee el archivo de entrada línea por línea y construye el Dataset.
//  Usamos move() al insertar cada punto para evitar copiar el vector
//  de doubles — con 1 millón de puntos, cada copia evitada importa.
//
//  Nota: no validamos que todas las líneas tengan la misma cantidad
//  de columnas porque asumimos que el CSV viene del notebook con
//  formato consistente. En producción eso debería verificarse.
// ─────────────────────────────────────────────
Dataset load_csv(const string &path)
{
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

        stringstream ss(line);
        string token;
        Point p;
        while (getline(ss, token, ','))
            p.push_back(stod(token));

        if (!p.empty())
            data.push_back(move(p));  // move evita copiar el vector interno
    }

    cout << "[IO] " << data.size() << " puntos, dim=" << data[0].size() << "\n";
    return data;
}

void save_labels(const string &path, const vector<int> &labels)
{
    ofstream f(path);
    for (int l : labels)
        f << l << "\n";
}

void save_centroids(const string &path, const vector<Centroid> &centroids)
{
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
//  sq_dist — distancia euclidiana AL CUADRADO
//
//  La razón de no calcular la raíz cuadrada:
//  Para comparar "¿cuál centroide está más cerca?" sólo importa
//  el orden relativo de las distancias, no su valor real.
//  Si d1² < d2², entonces d1 < d2. Calcular sqrt es una operación
//  costosa que se ejecutaría n*k veces por iteración — con n=1M y
//  k=8 son 8 millones de sqrts evitadas por iteración. Con ~50
//  iteraciones son 400 millones de sqrts ahorradas por ejecución.
//
//  El keyword inline le pide al compilador sustituir el cuerpo de
//  la función en cada llamada, eliminando el overhead del call stack.
//  Con -O2/-O3 el compilador lo haría de todas formas, pero ser
//  explícito comunica la intención al lector.
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
//  kmeanspp_init — inicialización K-Means++
//
//  Por qué no inicializar al azar:
//  Si ponemos k centroides uniformemente al azar, es probable que
//  varios caigan dentro del mismo cluster real. Eso obliga al algoritmo
//  a "repartirlos" durante las primeras iteraciones antes de que empiecen
//  a converger — puede duplicar o triplicar el número de iteraciones, o
//  quedar atrapado en un mínimo local malo.
//
//  K-Means++ evita eso sembrando los centroides con probabilidad
//  proporcional a D²(i): la distancia al cuadrado del punto i al
//  centroide ya elegido más cercano. Los puntos lejos de todos los
//  centroides existentes tienen más chance de ser elegidos, lo que
//  garantiza diversidad geográfica en la inicialización.
//
//  El costo extra es O(k * n) — pagado una sola vez antes del loop
//  principal. Vale la pena.
//
//  Por qué pasamos rng por referencia:
//  mt19937 tiene estado interno (~2.5 KB). Pasarlo por valor copiaría
//  ese estado y rompería la secuencia aleatoria. Con referencia, el
//  mismo generador avanza de forma continua y reproducible.
// ─────────────────────────────────────────────
vector<Centroid> kmeanspp_init(const Dataset &data, int k, mt19937 &rng)
{
    size_t n = data.size();
    vector<Centroid> centroids;
    centroids.reserve(k);  // evita reallocations durante el push_back del loop

    // Primer centroide: uniforme al azar entre todos los puntos
    uniform_int_distribution<size_t> unif(0, n - 1);
    centroids.push_back(data[unif(rng)]);

    // dist2[i] = distancia² mínima del punto i a cualquier centroide ya elegido.
    // Iniciamos con infinito para que la primera actualización siempre gane.
    vector<double> dist2(n, numeric_limits<double>::infinity());

    for (int c = 1; c < k; ++c)
    {
        // Truco de eficiencia: solo actualizamos dist2 contra el último
        // centroide elegido. Los candidatos anteriores ya están incorporados
        // en dist2 de iteraciones previas. Costo O(n) por centroide, no O(c*n).
        const Centroid &last = centroids.back();
        for (size_t i = 0; i < n; ++i)
        {
            double d = sq_dist(data[i], last);
            if (d < dist2[i])
                dist2[i] = d;
        }

        // discrete_distribution muestrea un índice con probabilidad proporcional
        // a los valores de dist2. Hace exactamente lo que describe el algoritmo
        // de K-Means++ sin implementar normalización ni búsqueda binaria a mano.
        discrete_distribution<size_t> weighted(dist2.begin(), dist2.end());
        centroids.push_back(data[weighted(rng)]);
    }
    return centroids;
}

// ─────────────────────────────────────────────
//  assign_clusters — Paso 2 (E-step)
//
//  "E" viene de Expectation del algoritmo EM (Expectation-Maximization).
//  Fijamos los centroides y buscamos la asignación óptima de puntos:
//  cada punto va al centroide más cercano.
//
//  Por qué retornamos el número de cambios en lugar de la inercia:
//  La inercia (suma de distancias²) es la métrica de convergencia de
//  sklearn, pero nuestro criterio es más estricto: paramos cuando
//  NINGÚN punto cambia de cluster. Esto da convergencia exacta, no
//  aproximada. El costo es que podemos tardar más iteraciones, pero
//  para el proyecto eso genera más trabajo paralelo — que es justamente
//  lo que queremos medir.
//
//  labels se pasa por referencia no-const porque lo modificamos in-place.
//  Retornar un vector nuevo significaría copiar n enteros por iteración.
// ─────────────────────────────────────────────
int assign_clusters(const Dataset &data,
                    const vector<Centroid> &centroids,
                    vector<int> &labels)
{
    int changes = 0;
    int k = static_cast<int>(centroids.size());

    for (size_t i = 0; i < data.size(); ++i)
    {
        // Inicializamos best_d con infinito para que cualquier distancia
        // real gane en la primera comparación del loop interno.
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

        // Solo incrementamos changes si el cluster realmente cambió.
        // Con esto evitamos sobrecontar en la primera iteración (donde
        // labels es todo -1) — aunque ahí todos cambiarían de todas formas.
        if (labels[i] != best_c)
        {
            labels[i] = best_c;
            ++changes;
        }
    }
    return changes;
}

// ─────────────────────────────────────────────
//  update_centroids — Paso 3 (M-step)
//
//  "M" viene de Maximization del algoritmo EM. Fijamos las asignaciones
//  y buscamos los centroides que minimizan la inercia — que son
//  exactamente las medias de cada cluster.
//
//  Por qué acumulamos sumas en lugar de calcular la media incremental:
//  Una media incremental requiere dividir n veces. Acumular sumas y
//  dividir al final requiere dividir k veces. Con n=1M y k=8 la
//  diferencia es de 125,000x.
//
//  El guard `if (counts[c] == 0) continue` previene división por cero
//  cuando un cluster queda vacío — puede pasar si la inicialización
//  puso dos centroides muy cerca. El centroide vacío se queda donde
//  está hasta que los movimientos de los otros le devuelvan puntos.
// ─────────────────────────────────────────────
void update_centroids(const Dataset &data,
                      const vector<int> &labels,
                      vector<Centroid> &centroids)
{
    int k = static_cast<int>(centroids.size());
    size_t dim = data[0].size();

    vector<Centroid> sums(k, Centroid(dim, 0.0));
    vector<int> counts(k, 0);

    for (size_t i = 0; i < data.size(); ++i)
    {
        int c = labels[i];
        for (size_t d = 0; d < dim; ++d)
            sums[c][d] += data[i][d];
        ++counts[c];
    }

    for (int c = 0; c < k; ++c)
    {
        if (counts[c] == 0)
            continue;
        for (size_t d = 0; d < dim; ++d)
            centroids[c][d] = sums[c][d] / counts[c];
    }
}

// ─────────────────────────────────────────────
//  kmeans_serial — driver principal del algoritmo
//
//  Orquesta las tres fases y mide el tiempo. Recibe Dataset& en lugar
//  de una ruta para que el benchmark pueda reutilizar los datos en
//  memoria sin recargar el CSV en cada repetición.
//
//  Por qué medimos el tiempo DENTRO del binario y no en Python:
//  Si Python midiera con time.perf_counter(), incluiría el tiempo de
//  fork del subproceso (~50ms en Linux) y la carga del CSV. Midiendo
//  desde C++ excluimos esas operaciones y obtenemos el tiempo real
//  del algoritmo. El benchmark parsea la línea "tiempo=X.XXXXs" del
//  stdout para extraer ese valor.
//
//  Por qué seed como parámetro:
//  Para que el benchmark pueda dar la misma semilla a serial y paralelo,
//  garantizando que la diferencia de tiempo no venga de distintas
//  inicializaciones sino solo del paralelismo.
// ─────────────────────────────────────────────
double kmeans_serial(const Dataset &data, int k,
                     int max_iter = 300, unsigned seed = 42,
                     bool verbose = true,
                     bool save = true)
{
    size_t n = data.size();
    mt19937 rng(seed);

    auto centroids = kmeanspp_init(data, k, rng);

    // Inicializamos labels a -1 ("sin asignar") para que en la primera
    // iteración todos los puntos cuenten como cambios y no se active
    // accidentalmente el criterio de convergencia.
    vector<int> labels(n, -1);

    // El reloj empieza DESPUÉS de la inicialización K-Means++.
    // La versión paralela hace lo mismo, así que la comparación es justa:
    // medimos solo el loop de Lloyd, no el setup.
    auto t0 = chrono::high_resolution_clock::now();

    int iter = 0;
    for (iter = 1; iter <= max_iter; ++iter)
    {
        int changes = assign_clusters(data, centroids, labels);
        update_centroids(data, labels, centroids);

        if (verbose && (iter == 1 || iter % 10 == 0))
            cout << "  iter=" << setw(3) << iter
                 << "  cambios=" << setw(7) << changes << "\n";

        // Convergencia exacta: paramos cuando ningún punto cambió de cluster.
        // max_iter=300 actúa como safety net — en la práctica, con K-Means++
        // y datos razonables siempre converge bien antes de ese límite.
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
//
//  Mínimo: parsear args, cargar datos, ejecutar.
//  La lógica está en kmeans_serial(), no aquí.
//
//  Usamos argc/argv en lugar de cin para poder llamar el binario
//  desde Python con subprocess.run([binary, csv, k, ...]) sin
//  necesidad de pipes ni stdin.
// ─────────────────────────────────────────────
int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cerr << "Uso: " << argv[0]
             << " <csv_file> <k> [max_iter=300] [seed=42]\n";
        return 1;
    }

    string   csv      = argv[1];
    int      k        = stoi(argv[2]);
    int      max_iter = (argc > 3) ? stoi(argv[3]) : 300;

    // seed es unsigned porque mt19937 toma unsigned. stoul en lugar de
    // stoi evita comportamiento indefinido si alguien pasa un número
    // mayor a INT_MAX como semilla.
    unsigned seed = (argc > 4) ? stoul(argv[4]) : 42;

    cout << "══════════════════════════════════════════\n";
    cout << " K-Means Serial  k=" << k
         << "  max_iter=" << max_iter << "\n";
    cout << "══════════════════════════════════════════\n";

    Dataset data = load_csv(csv);
    kmeans_serial(data, k, max_iter, seed, true, true);

    return 0;
}