#include "headers.hpp"

extern int myRank;
extern int nbTasks;

//================================================================================
// Build the local split for MPI communications
//================================================================================

void buildLocSplitMPI(Problem& q, int lrep, int crep)
{
  int ROWS = q.A.rows(), COLS = q.A.cols(), NPROWS = 1, NPCOLS = 1, Nc = 0;
  double a[ROWS*COLS];
  if (lrep == 1 && crep == 0)
  NPROWS = nbTasks;
  else if (lrep == 0 && crep == 1)
    NPCOLS = nbTasks;
  else if (lrep == 1 && crep == 1)
  {
    NPROWS = nbTasks/sqrt(nbTasks);
    NPCOLS = NPROWS;
  }

  int BLOCKROWS = ROWS/NPROWS;  /* number of rows in _block_ */
  int BLOCKCOLS = COLS/NPCOLS; /* number of cols in _block_ */
  int n1 = 0, n2 = 0;
  MPI_Allreduce(&BLOCKROWS, &n1, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&BLOCKCOLS, &n2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (myRank == nbTasks-1 && lrep == 1 && crep == 0)
    BLOCKROWS = BLOCKROWS + ROWS - n1;
  if (myRank == nbTasks-1 && lrep == 0 && crep == 1)
    BLOCKCOLS = BLOCKCOLS + COLS - n2;
  if (myRank == nbTasks-1 && lrep == 1 && crep == 1)
  {
    BLOCKROWS = BLOCKROWS + ROWS - n1;
    BLOCKCOLS = BLOCKCOLS + COLS - n2;
  }
  if (nbTasks != NPROWS*NPCOLS) 
  {
    printf("Error: number of PEs %d != %d x %d\n", nbTasks, NPROWS, NPCOLS);
    MPI_Finalize();
    exit(-1);
  }

  if (myRank == 0) 
  {
    int ii = 0;
    for (int j = 0; j < COLS; ++j)
      for (int i = 0; i < ROWS; ++i)
      {
        a[ii] = q.A(i,j);
        ii++;
      }
  }

  double b[BLOCKROWS*BLOCKCOLS];
  for (int ii=0; ii<BLOCKROWS*BLOCKCOLS; ii++) b[ii] = 0;

  MPI_Datatype blocktype;
  MPI_Datatype blocktype2;

  MPI_Type_vector(BLOCKROWS, BLOCKCOLS, COLS, MPI_DOUBLE, &blocktype2);
  MPI_Type_create_resized( blocktype2, 0, sizeof(double), &blocktype);
  MPI_Type_commit(&blocktype);

  int disps[NPROWS*NPCOLS];
  int counts[NPROWS*NPCOLS];
  for (int ii=0; ii<NPROWS; ii++) {
    for (int jj=0; jj<NPCOLS; jj++) {
      disps[ii*NPCOLS+jj] = ii*COLS*BLOCKROWS+jj*BLOCKCOLS;
      counts [ii*NPCOLS+jj] = 1;
    }
  }
  MPI_Scatterv(a, counts, disps, blocktype, b, BLOCKROWS*BLOCKCOLS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  int ii = 0;
  for (int i = 0; i < BLOCKROWS; ++i)
    for (int j = 0; j < BLOCKCOLS; ++j)
    {
      q.A(i,j) = b[ii];
      ii++;
    }
  int NPv = NPCOLS, NPblock = BLOCKCOLS;
  if (lrep == 1 && crep == 0)
  {
    NPv = NPROWS;
    NPblock = BLOCKROWS;
  }

  Vector z = Vector::Zero(NPblock);
  MPI_Scatter(q.v.data(), NPblock, MPI_DOUBLE, z.data(), NPblock, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  q.v = z;
  
  q.A.conservativeResize(BLOCKROWS,BLOCKCOLS);

  for (int i = 0; i < q.A.rows(); ++i)
    for (int j = 0; j < q.A.cols(); ++j)
      if (q.A(i,j) < 1.e-250 && q.A(i,j) > -1.e-250)
        q.A(i,j) = 0;

}

Vector MyGatherv (Vector& v)
{
  int *counts, *displacements;
  counts = (int *) malloc(nbTasks*sizeof(int));
  displacements = (int *) malloc(nbTasks*sizeof(int));
  int n = 0, rows = v.rows();
  for(int nTask = 0; nTask < nbTasks; ++nTask)
  {
    counts[nTask] = rows;
    displacements[nTask] = rows*nTask;
    if (nTask == nbTasks - 1)
      displacements[nTask] = counts[nTask-1]*nTask;
  }
  MPI_Allreduce(&rows, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  Vector vglob = Vector::Zero(n);
  MPI_Allgatherv(v.data(), v.rows(), MPI_DOUBLE, vglob.data(), counts, displacements, MPI_DOUBLE, MPI_COMM_WORLD);

  delete[] counts;
  delete[] displacements;

  return vglob;
}