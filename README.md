# K-Means Benchmark: Serial vs Parallel (OpenMP)

**Proyecto Apertura — Cómputo Paralelo y en la Nube, ITAM 2026**

This project benchmarks the performance of K-Means clustering algorithms in serial and parallel implementations using OpenMP. It compares execution times across different data sizes, dimensions (2D and 3D), and thread counts to demonstrate speedup from parallelization.

## Features

- **Data Generation**: Automatically generates synthetic datasets using scikit-learn's `make_blobs` for consistent benchmarking.
- **Serial Implementation**: Pure C++ serial K-Means.
- **Parallel Implementation**: OpenMP-parallelized K-Means with configurable thread counts.
- **Comprehensive Benchmarking**: Tests multiple data sizes (100k to 1M points), dimensions (2D/3D), and thread configurations (1, cores/2, cores, cores*2).
- **Statistical Analysis**: Runs multiple repetitions and computes mean times and speedups.
- **Visualization**: Generates speedup plots using matplotlib (if available).
- **Cross-Platform**: Designed to work on Windows and Unix-like systems.

## Project Structure

```
.
├── benchmark.py          # Main Python script for benchmarking
├── kmeans_serial.cpp     # Serial K-Means implementation
├── kmeans_parallel.cpp   # Parallel K-Means with OpenMP
├── requirements.txt      # Python dependencies
├── setup.bat             # Windows setup script (creates venv, installs deps, compiles C++)
├── README.md             # This file
└── data/                 # Generated datasets (created automatically)
    └── results.csv       # Benchmark results (generated)
    └── *.png             # Speedup plots (generated)
```

## Requirements

### System Requirements
- **C++ Compiler**: GCC with OpenMP support (e.g., g++ from MinGW or LLVM on Windows)
- **Python**: 3.8+ (tested with 3.13)
- **Operating System**: Windows, Linux, or macOS

### Python Dependencies
- numpy
- pandas
- scikit-learn
- matplotlib (optional, for plotting)

## Setup

### Quick Setup (Windows)
1. Ensure you have Python and g++ installed.
2. Run the setup script:
   ```
   setup.bat
   ```
   This will:
   - Create a virtual environment (`.venv`)
   - Install Python dependencies
   - Compile the C++ binaries

### Manual Setup
1. Create a virtual environment:
   ```
   python -m venv .venv
   ```
2. Activate the environment:
   - Windows: `.venv\Scripts\activate`
   - Unix: `source .venv/bin/activate`
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Compile C++ binaries:
   ```
   g++ -O2 -std=c++17 -o kmeans_serial kmeans_serial.cpp
   g++ -O2 -std=c++17 -fopenmp -o kmeans_parallel kmeans_parallel.cpp
   ```

## Usage

After setup, run the benchmark:

```
python benchmark.py
```

### Command-Line Options
- `--k K`: Number of clusters (default: 8)
- `--reps N`: Number of repetitions per configuration (default: 10)
- `--max_iter N`: Maximum iterations for K-Means (default: 300)
- `--seed S`: Random seed for reproducibility (default: 42)
- `--sizes S1,S2,...`: Comma-separated list of data sizes (default: 100000,200000,300000,400000,600000,800000,1000000)
- `--input_csv FILE`: Use external CSV file instead of generating synthetic data

Examples:
```
# Quick test with 3 reps and smaller datasets
python benchmark.py --reps 3 --sizes 100000,200000

# Custom clusters and iterations
python benchmark.py --k 10 --max_iter 500 --reps 5

# Different random seed for variety
python benchmark.py --seed 123 --reps 3

# Use external CSV
python benchmark.py --input_csv data.csv
```

## Output

The script generates:
- **results.csv**: Raw benchmark data (times for each run)
- **speedup_2d.png**: Speedup plot for 2D data
- **speedup_3d.png**: Speedup plot for 3D data
- **speedup_combined.png**: Combined 2D/3D speedup plot

Console output shows progress and summary statistics (mean time, standard deviation, speedup).

## Benchmark Configuration

All parameters are configurable via command-line arguments:

- **Clusters (k)**: 8 (use `--k` to change)
- **Max Iterations**: 300 (use `--max_iter` to change)
- **Data Sizes**: 100k, 200k, 300k, 400k, 600k, 800k, 1M points (use `--sizes` to change)
- **Dimensions**: 2D and 3D (auto-detected from data)
- **Threads**: 1, cores/2, cores, cores*2 (detected automatically)
- **Repetitions**: 10 per configuration (use `--reps` to change)
- **Random Seed**: 42 (use `--seed` to change for reproducibility)

## Implementation Details

### Serial Version (`kmeans_serial.cpp`)
- Standard Lloyd's algorithm
- Single-threaded execution

### Parallel Version (`kmeans_parallel.cpp`)
- OpenMP parallelization
- Parallel loops for distance calculations and centroid updates
- Thread-safe centroid accumulation

### Python Script (`benchmark.py`)
- Manages data generation and execution
- Handles subprocess calls to C++ binaries
- Computes statistics and generates plots
- Automatically uses virtual environment Python

## Troubleshooting

- **"Module not found"**: Ensure virtual environment is activated or run `python benchmark.py` (auto-detection handles venv)
- **"Binary not found"**: Check that C++ binaries compiled successfully
- **Compilation errors**: Ensure g++ supports OpenMP (`-fopenmp` flag)
- **No plots**: Install matplotlib or ignore warnings (plots are optional)

## Performance Notes

- Parallel speedup depends on CPU cores and data size
- Larger datasets show better parallel scaling
- OpenMP overhead may reduce speedup for small datasets or high thread counts

## License

Academic project for ITAM 2026 course.</content>
<parameter name="filePath">c:\cpp\Saul\README.md
