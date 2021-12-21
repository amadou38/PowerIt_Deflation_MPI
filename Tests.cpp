#include "headers.hpp"

extern int myRank;
extern int nbTasks;


void TestBLAS(Problem& q)
{
	Problem p = q;
    // cout << "hehehe" << endl;
	buildLocSplitMPI(p, 0, 1);
    double t1 = MPI_Wtime();
    double a = BLAS1(p.v, p.v);
    double t2 = MPI_Wtime();
    Vector s = BLAS2(p.A, p.v);
    double t3 = MPI_Wtime();
    // Vector ss = BLAS3(p.A, p.v);
    double t4 = MPI_Wtime();
    
    for (int proc=0; proc<nbTasks; proc++) {
        if (myRank == proc) {
            cout << "myRank = " << myRank << endl;
            // if (myRank == 0)
            //     cout << "Global matrix: \n" << q.A << endl;
            // cout << "Local Matrix:\n" << p.A << endl;
            // cout << "Local Vector:\n" << p.v << endl;
            // cout << "a = " << a << endl << endl;
            cout << "s:\n" << s << endl << endl;
            // cout << "ss:\n" << ss << endl << endl;
        }
    }


    // if (myRank == 0)
	   //  cout << "\n\nBLAS1 runtime: " << t2-t1 << " sec\n\nBLAS2 runtime: " << t3-t2 << " sec\n\nBLAS3 runtime: " << t4-t3 << " sec\n\n";
}

void TestERAM(Problem& p, Vector& LmbdApp, Matrix& Vm, int m, int jj, double tol, int maxiter)
{
	double t1 = MPI_Wtime();
	ERAM(p, LmbdApp, Vm, m, jj, maxiter, tol);
    double t = MPI_Wtime() - t1;
    sortVect(p.Lmbd);
    if (myRank == 0)
    {
        // cout << "Data present in the file: " << endl << endl;
        // cout << "Matrix p.A: \n" << p.A << endl << endl;
        // cout << "Initial vector p.v: \n" << p.v << endl << endl;
    
        cout << "Exact Eigenvalues: " << endl;
        for (int i = 0; i < p.Lmbd.rows(); i++)
            cout << p.Lmbd(i) << "  ";
        cout << endl;
        cout << "Approximate Eigenvalues: " << endl;
        for (int i = 0; i < LmbdApp.rows(); i++)
            cout << LmbdApp(i) << "  ";
        cout << endl;
        cout << "Approximate Eigenvectors: \n" << Vm << endl;
    
    	cout << "\n\nERAM runtime: " << t << " sec\n\n";
    
    }
}