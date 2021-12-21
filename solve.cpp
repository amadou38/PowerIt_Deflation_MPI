#include "headers.hpp"

extern int myRank;
extern int nbTasks;

void PuissanceIt(Problem& p, double tol)
{
	p.v = p.v/sqrt(BLAS1(p.v,p.v));
	Vector y = BLAS2(p.A,p.v);
	Vector x = y/sqrt(BLAS1(y,y));
	double err = abs(abs(BLAS1(p.v,x)) - 1);
	while (err > tol)
	{
		p.v = x;
		y = BLAS2(p.A,p.v);
		x = y/sqrt(BLAS1(y,y));
		err = abs(abs(BLAS1(p.v,x)) - 1);
	}
	p.lambda = y(0)/p.v(0);
}

void Deflation(Problem& p, Matrix& B, int m, double tol)
{
	Vector v = p.v;
	B = Matrix::Zero(p.v.rows()*nbTasks, m);
	
	p.Lmbd = Vector::Zero(m);

	PuissanceIt(p, tol);
	p.Lmbd(0) = p.lambda;
	Vector vglob = MyGatherv(p.v);
	for (int i = 0; i < vglob.rows(); ++i)
		B(i,0) = vglob(i);

	for (int k = 1; k < m; ++k)
	{
		Matrix M = Prodvc(p.v, p.v);
		p.A = p.A - p.lambda*M;
		p.v = v;
		PuissanceIt(p, tol);
		p.Lmbd(k) = p.lambda;
		vglob = MyGatherv(p.v);
		for (int i = 0; i < vglob.rows(); ++i)
			B(i,k) = vglob(i);		
	}
}

void ERAM(Problem& p, Vector& LmbdApp, Matrix& Vm, int m, int jj, int maxiter, double tol)
{
	double hm1 = 0;
  	int n = p.A.rows(), cnt = 0;
  	Matrix Hm, Q(n,n), R(n,n);
  	Vector vm1, Err = Vector::Zero(m);
  	Eigen::MatrixXd D, V;
	
	for (int s = 1; s <= m/2; s++)
  	{
  		double maxErr = 1;
  		int iter = 0;
	  	while(iter < maxiter && maxErr > tol)
	  	{
  			Problem q = p;

	  		Arnoldi(q, Vm, Hm, &hm1, vm1, s, jj);
	  		MGS(Vm, Q, R);
	  		Eigen::EigenSolver<Eigen::MatrixXd> es(Hm);
	  		D = es.pseudoEigenvalueMatrix();
	  		V = es.pseudoEigenvectors();
	  		Matrix S = (Matrix)V;
	  		sort(D,V);
	  		for (int i = 0; i < s; ++i)
	  			Err(i) = hm1*fabs(S(s-1,i));
	  		sortVect(Err);
	  		maxErr = Err(0);
	  		Vector Alpha = Vector::Random(s);
	  		Vm = Q*V;
	  		p.v = Vector::Zero(n);
	  		for (int i = 0; i < n; ++i)
	  			for (int j = 0; j < s; ++j)
			  		p.v(i) += /*Alpha(j)**/Vm(i,j);
			iter++;
		}
		LmbdApp = Vector::Zero(D.rows());
		for (int i = 0; i < D.rows(); ++i)
			LmbdApp(i) = D(i,i);

		if (maxErr < tol)
			break;
  	}
  	
  OrthogVerif(Vm);

}

void Arnoldi(Problem& p, Matrix& Vm, Matrix& Hm, double *hm1, Vector& vm1, int m, int jj)
{
	int n = p.A.rows();
	
	Vm.resize(n,m);
	Hm.resize(m,m);
	vm1.resize(n);
	Vm = Matrix::Zero(n,m);
	Hm = Matrix::Zero(m,m);
	vm1 = Vector::Zero(n);
	double h = 0;
	Vector x = Vector::Zero(n);
	vm1 = p.v/sqrt(Prodsc(p.v,p.v));
	for (int j = 0; j < m; ++j)
	{
		for (int k = 0; k < n; ++k)
			Vm(k,j) = vm1(k);
		vm1 = p.A*vm1;
		for (int i = 0; i <= j; ++i)
		{
			for (int k = 0; k < n; ++k)
				x(k) = Vm(k,i);
			Hm(i,j) = Prodsc(vm1,x);
			vm1 = vm1 - x*Hm(i,j);
		}
		h = sqrt(Prodsc(vm1,vm1));
		if (h == 0)
		{
			cout << "Le sous-espace v1...v" << j+1 << " est invariant !!!" << endl;
			break;
		}
		vm1 = vm1/h;
		if (j+1 != m)
			Hm(j+1,j) = h;
	}
	*hm1 = h;
}

void GS(Matrix& A, Matrix& Q, Matrix& R)
{
	int m  = A.rows(), n = A.cols();
	Vector w(m), q(m);
	Q.resize(m,n);
	R.resize(m,n);
	for (int k = 0; k < n; ++k)
	{
		for (int j = 0; j < m; ++j)
			w(j) = A(j,k);
		
		for (int j = 0; j <= k-1; ++j)
		{
			for (int l = 0; l < m; ++l)
				q(l) = Q(l,j);
			R(j,k) = Prodsc(q,w);
		}
		for (int j = 0; j <= k-1; ++j)
		{
			for (int l = 0; l < m; ++l)
				q(l) = Q(l,j);
			w = w - q*R(j,k);
		}
		R(k,k) = sqrt(Prodsc(w,w));
		q = w/R(k,k);
		for (int j = 0; j < m; ++j)
			Q(j,k) = q(j);
	}
}

void MGS(Matrix& A, Matrix& Q, Matrix& R)
{
	int m  = A.rows(), n = A.cols();
	Vector w(m), q(m);
	Q.resize(m,n);
	R.resize(m,n);
	for (int k = 0; k < n; ++k)
	{
		for (int j = 0; j < m; ++j)
			w(j) = A(j,k);
		
		for (int j = 0; j <= k-1; ++j)
		{
			for (int l = 0; l < m; ++l)
				q(l) = Q(l,j);
			R(j,k) = Prodsc(q,w);
			w = w - q*R(j,k);
		}
		R(k,k) = sqrt(Prodsc(w,w));
		q = w/R(k,k);
		for (int j = 0; j < m; ++j)
			Q(j,k) = q(j);
	}
}