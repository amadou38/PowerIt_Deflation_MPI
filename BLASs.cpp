#include "headers.hpp"

extern int myRank;
extern int nbTasks;

double BLAS1(Vector a, Vector b)
{
	double pscLoc = 0, pscGlob = 0;
    
  for(int i=0; i<a.rows(); i++)
  		pscLoc += a(i)*b(i);

  MPI_Allreduce(&pscLoc, &pscGlob, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  
  return pscGlob;
}

Vector BLAS2(Matrix A, Vector x)
{
	Vector pMVGlob = Vector::Zero(A.rows());
  	Vector a = Vector::Zero(x.rows());

  	for (int i = 0; i < A.rows(); ++i)
  	{
  		for (int j = 0; j < x.rows(); ++j)
  			a(j) = A(i,j);
  		pMVGlob(i) = BLAS1(a, x);
  	}

  int size = x.rows();
  Vector z = Vector::Zero(size);
  MPI_Scatter(pMVGlob.data(), size, MPI_DOUBLE, z.data(), size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  pMVGlob.resize(z.rows());
  pMVGlob = z;
  return pMVGlob;
}

Vector BLAS3(Matrix A, Vector x)
{

	Vector pMVGlob = BLAS2(A, x);
	
  pMVGlob = BLAS2(A, pMVGlob) + pMVGlob + x;

  pMVGlob.conservativeResize(x.rows());

  return pMVGlob;
}