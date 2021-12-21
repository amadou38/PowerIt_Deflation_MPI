#ifndef HEADERS_HPP
#define HEADERS_HPP

#include <iostream>
#include <fstream>
#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>
#include <mpi.h>
#include <math.h>
#include <stdio.h>

using namespace std;

//================================================================================
// SPECIAL TYPES
//================================================================================

// Types for dense storage
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<double, 1, Eigen::Dynamic> LineVect;
typedef Eigen::Matrix<int,    Eigen::Dynamic, 1> IntVector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Matrix<int,    Eigen::Dynamic, Eigen::Dynamic> IntMatrix;

// Structure for problem
struct Problem
{
   Matrix A;
   Vector v;
   double lambda;
   Vector Lmbd;
};

  

//================================================================================
// FUNCTIONS
//================================================================================

//==== Functions in 'BLASs.cpp'

double BLAS1(Vector a, Vector b);
Vector BLAS2(Matrix A, Vector x);
Vector BLAS3(Matrix A, Vector x);

//==== Functions in 'Tests.cpp'

void TestBLAS(Problem& q);
void TestERAM(Problem& p, Vector& LmbdApp, Matrix& Vm, int m, int jj, double tol, int maxiter);

//==== Functions in 'Datastruct.cpp'

// Read/Write of the data from datafiles
void readData(Problem& p, string filename);
void writeData(Problem& p, string filename);

//==== Functions in 'parallel.cpp'

void buildLocSplitMPI(Problem& q, int lrep, int crep);
Vector MyGatherv (Vector& v);

//==== Function in 'solve.cpp'

void PuissanceIt(Problem& p, double tol);
void Deflation(Problem& p, Matrix& B, int m, double tol);
void ERAM(Problem& p, Vector& LmbdApp, Matrix& Vm, int m, int jj, int maxiter, double tol);
void Arnoldi(Problem& p, Matrix& Vm, Matrix& Hm, double *hm1, Vector& vm1, int m, int jj);


// Gram-Schmidt function 
void GS(Matrix& A, Matrix& Q, Matrix& R);
void MGS(Matrix& A, Matrix& Q, Matrix& R);

//==== Functions in 'accessories.cpp'

Matrix Prodvc(Vector a, Vector b);
void sortVect(Vector& v);
void sort(Eigen::MatrixXd& Eigval, Eigen::MatrixXd& Eigvect);
double Prodsc(Vector a, Vector b);
void OrthogVerif(Matrix& Q);
void MatrixRows(Vector& v, Matrix& A, int i);
void MatrixCols(Vector& v, Matrix& A, int j);
void Tests(Problem& p, int cas, int dim);
void Block(Matrix& AA, Matrix& B, int i1, int i2, int j1, int j2);
double min(double a, double b);

#endif /* HEADERS_HPP */