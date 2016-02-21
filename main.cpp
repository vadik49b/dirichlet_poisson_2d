#include <iostream>
#include <omp.h>
#include <cmath>

// config
#define OMP_THREADS_NUM 4

// solution globals
#define Dx 1
#define Dy 2
#define N 10
#define h ((double) Dx / N)
#define Nx ((int) (Dx / h) + 1)
#define Ny ((int) (Dy / h) + 1)
#define Rit 10000

// tiling
#define r1 4
#define r2 4

using namespace std;

double f(int i, int j) {
    return 2 * (pow(i * h, 2) + pow(j * h, 2));
}

double exactValue(int i, int j) {
    return pow(i * h, 2) * pow(j * h, 2);
}

void logSolutionError(double **u) {
    double maxError = u[0][0];
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            double err = abs(u[i][j] - exactValue(i, j));
            cout << err << " ";
            maxError = max(maxError, err);
        }
        cout << endl;
    }
    cout << "\nsolution max error: " << maxError << endl;
}

double** allocateAndFillMatrix() {
    double** ary = new double*[Nx];
    for (int i = 0; i < Nx; ++i) {
        ary[i] = new double[Ny];
        for (int j = 0; j < Ny; ++j) {
            ary[i][j] = 0;
        }
    }
    for (int i = 0; i < Nx; ++i) {
        ary[i][Ny - 1] = exactValue(i, Ny - 1);
    }
    for (int j = 0; j < Ny; ++j) {
        ary[Nx - 1][j] = exactValue(Nx - 1, j);
    }
    return ary;
}


void logMatrix(double** m) {
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            cout << m[i][j] << " ";
        }
        cout << endl;
    }
}

void seidel(double** u, int i, int j) {
    u[i][j] = 0.25 * (u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1] - pow(h, 2) * f(i, j));
}

void solveSimple(double** u) {
    for (int it = 0; it < Rit; ++it) {
        for (int i = 1; i < Nx - 1; ++i) {
            for (int j = 1; j < Ny - 1; ++j) {
                seidel(u, i, j);
            }
        }
    }
}

void tile(double** u, int ig, int jg) {
    for (int i = 1 + ig * r1; i < min((ig + 1) * r1 + 1, Nx - 1); ++i) {
        for (int j = 1 + jg * r2; j < min((jg + 1) * r2 + 1, Ny - 1); ++j) {
            seidel(u, i, j);
        }
    }
}

void solveSimpleTiling(double** u) {
    int Q1 = (int) ceil(((double) Nx / r1));
    int Q2 = (int) ceil(((double) Ny / r2));

    for (int it = 0; it < Rit; ++it) {
        for (int ig = 0; ig < Q1; ++ig) {
            for (int jg = 0; jg < Q2; ++jg) {
                tile(u, ig, jg);
            }
        }
    }
}

int main() {
    omp_set_num_threads(OMP_THREADS_NUM);

    double runtime = omp_get_wtime();

    double** u = allocateAndFillMatrix();

//    solveSimple(u);
    solveSimpleTiling(u);
    logSolutionError(u);

    cout << "\nruntime was: " << omp_get_wtime() - runtime << "\n";

    return 0;
}