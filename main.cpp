#include <iostream>
#include <fstream>
#include <omp.h>
#include <cmath>
#include <stdio.h>
#include <vector>
#include <mutex>
using namespace std;

// config
#define OMP_THREADS_NUM 4

// solution globals
#define Dx 1
#define Dy 2
#define N 100
#define h ((double) Dx / N)
#define Nx ((int) (Dx / h) + 1)
#define Ny ((int) (Dy / h) + 1)
#define Rit 6

// tiling
#define r1 2
#define r2 40
#define r3 40
#define Mover 1

struct LOG_TICK {
	double time;
	int i;
	int j;
};

mutex logMutex;
vector <LOG_TICK> LOG_BUF;



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
            maxError = max(maxError, err);
        }
    }
    cout << "solution max error: " << maxError << "\n========\n";
}

void logMatrix(double **u) {
	cout << "U: " << "\n========\n";
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            cout << u[i][j] << " ";
        }
        cout << endl;
    }
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


void seidel(double** u, int i, int j) {

    // logging
	logMutex.lock();
	LOG_TICK tick;
	tick.i = i;
	tick.j = j;
	LOG_BUF.push_back(tick);
	logMutex.unlock();

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

void solveSimpleParallel(double** u) {
    #pragma omp parallel
    {
        for (int it = 0; it < Rit; ++it) {
            #pragma omp for
            for (int i = 1; i < Nx - 1; ++i) {
                for (int j = 1; j < Ny - 1; ++j) {
                    seidel(u, i, j);
                }
            }
        }
    }

}

void tile(double** u, int ig, int jg) {
    for (int i = 1 + ig * r2; i < min((ig + 1) * r2 + 1, Nx - 1); ++i) {
        for (int j = 1 + jg * r3; j < min((jg + 1) * r3 + 1, Ny - 1); ++j) {
            seidel(u, i, j);
        }
    }
}

void solveSimpleTiling(double** u) {
    int Q1 = (int) ceil(((double) Nx / r2));
    int Q2 = (int) ceil(((double) Ny / r3));
    #pragma omp parallel
    {
        for (int it = 0; it < Rit; ++it) {
            #pragma omp for
            for (int ig = 0; ig < Q1; ++ig) {
                for (int jg = 0; jg < Q2; ++jg) {
                    tile(u, ig, jg);
                }
            }
        }
    }
}

void haloTile(double** u, int ig, int jg, int lg) {
    for (int l = 1 + lg * r1; l < min((lg + 1) * r1 + 1, Rit + 1); ++l) {
        for (int i = max(l - (lg + 1) * r1 + ig * r2 + 1, 1); i < min(-l + (lg + 1) * r1 + (ig + 1) * r2 + 1, Nx - 1); ++i) {
            for (int j = max(l - (lg + 1) * r1 + jg * r3 + 1, 1); j < min(-l + (lg + 1) * r1 + (jg + 1) * r3 + 1, Ny - 1); ++j) {
                seidel(u, i, j);
            }
        }
    }
}

void solve3dTiling(double** u) {
    int Q1 = (int) ceil(((double) Rit / r1));
    int Q2 = (int) ceil(((double) Nx / r2));
    int Q3 = (int) ceil(((double) Ny / r3));

    #pragma omp parallel
    {
    	for (int lg = 0; lg < Q1; ++lg) {
    		#pragma omp for
	        for (int ig = 0; ig < Q2; ++ig) {
	            for (int jg = 0; jg < Q3; ++jg) {
	            	// printf("g %d %d %d\n", lg, ig, jg);
	                haloTile(u, ig, jg, lg);
	            }
	        }
    	}
    }
}

void showRuntime(double runtime) {
    double currentTime = omp_get_wtime();
    cout << "runtime was: " << currentTime - runtime << endl;
}

int main() {
    omp_set_num_threads(OMP_THREADS_NUM);

    cout << "Nx: " << Nx << ", Ny: " << Ny << ", h: " << h << ", Rit: " << Rit << "\n_______\n";

    // cout << "simple:\n";
    double** u = allocateAndFillMatrix();
    double runtime = omp_get_wtime();
    // solveSimple(u);
    // showRuntime(runtime);
    // logSolutionError(u);

    cout << "simple parallel:\n";
    u = allocateAndFillMatrix();
    runtime = omp_get_wtime();
    solveSimpleParallel(u);
    showRuntime(runtime);
    // logSolutionError(u);

    // cout << "simple parallel tiling:\n";
    // u = allocateAndFillMatrix();
    // runtime = omp_get_wtime();
    // solveSimpleTiling(u);
    // showRuntime(runtime);
    // logSolutionError(u);
    
    // cout << "3d test parallel tiling:\n";
    // u = allocateAndFillMatrix();
    // runtime = omp_get_wtime();
    // solve3dTiling(u);
    // showRuntime(runtime);		
    // logSolutionError(u);

    ofstream logfile("log.txt");
	for (vector<LOG_TICK>::iterator it = LOG_BUF.begin() ; it != LOG_BUF.end(); ++it) {
		logfile << ' ' << (*it).i << ' ' << (*it).j << endl;
	}
	logfile.close();

    return 0;
}