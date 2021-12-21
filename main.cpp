#include "headers.hpp"

int myRank;
int nbTasks;

int main(int argc, char *argv[])
{
	cout.precision(2);

    // 1. Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nbTasks);

	Problem p;

    int n = 96;   // taille de la matrice

    Tests(p, 3, n);
    Matrix A = p.A;
    Vector v = p.v;
    buildLocSplitMPI(p, 0, 1);

    int m = 6;
    double tol = 1e-15;
    Matrix B;
    Vector Lambda;

    sortVect(p.Lmbd);
    if (myRank == 0)
    {
        cout << "Data present in the file (size: " << n << "x" << n << "): " << endl << endl;
        cout << "==== proc 0-" << nbTasks-1 << ": size of A: " << p.A.rows() << "x" << p.A.cols() << " and size of v: " << p.v.rows() << " ======\n";
        cout << "==== proc " << nbTasks << ": size of A: " << A.rows()*nbTasks - p.A.rows()*(nbTasks-1) << "x" << A.cols() - p.A.cols()*(nbTasks-1) << " and size of v: " << v.rows() - p.v.rows()*(nbTasks-1) << " ======\n\n";
        // cout << "Matrix p.A (global): \n" << A << endl << endl;
        // cout << "Initial vector p.v (global): \n" << v << endl << endl;
        cout << "\nExact Eigenvalues : " << endl;
        for (int i = 0; i < 10/*p.Lmbd.rows()*/; i++)
            cout << p.Lmbd(i) << "  ";
        cout << endl;
    }

    double t1 = MPI_Wtime();
    // Deflation method (with Puissance It)
    Deflation(p, B, m, tol);
    double t2 = MPI_Wtime();

    if (myRank == 0)
    {
        cout << "Approximate Eigenvalues (for Deflation): " << endl;
        for (int i = 0; i < m; i++)
            cout << p.Lmbd(i) << "  ";
        // cout << "\nAnd associated eigenvectors (global):\n" << B << endl << endl;

        cout << "\n\nDebflation (with Puissance It) runtime: " << t2-t1 << " sec\n\n";
    }


    // 3. Finilize MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

	return 0;
}