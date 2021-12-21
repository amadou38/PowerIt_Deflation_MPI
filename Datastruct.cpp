#include "headers.hpp"

extern int myRank;
extern int nbTasks;

void readData(Problem& p, string filename) 
{
  string line;
  int n, m;
  ifstream datafile(filename.c_str());
  if (!datafile)
    printf("ERROR: Unable to open file named '%s'.\n", filename.c_str()); 
  else
  {
    while (datafile.good())
    {
      getline (datafile,line);
      if (line.compare("$BeginData") == 0)
        datafile >> n >> m;
      
      p.A.resize(n,m);
      p.v.resize(n);
      if (line.compare("$BeginMatrix") == 0)
      {
        for (int i = 0; i < n; ++i)
          for (int j = 0; j < m; ++j)
            datafile >> p.A(i,j);
      }
      
      if (line.compare("$BeginVector") == 0)
      {
        for (int i = 0; i < n; ++i)
          datafile >> p.v(i);
      }
      
    }
  }
  datafile.close();
}

void writeData(Problem& p, string filename)
{
  ofstream datafile(filename.c_str());
  
  datafile << "------------------------------------------------------\n";
  datafile << "|\n|\n|\n|\n|\n";
  datafile << "-------------------------------------------------------\n\n";
  datafile << "$BeginData" << endl;
  datafile << p.A.rows() << " " << p.A.cols() << endl;
  datafile << "$EndData" << endl;

  datafile << "$BeginMatrix" << endl;
  for (int i = 0; i < p.A.rows(); ++i)
  {
    for (int j = 0; j < p.A.cols(); ++j)
      datafile << /*setprecision(2) <<*/ p.A(i,j) << " ";
    datafile << endl;
  }
  datafile << "$EndMatrix" << endl;

  datafile << "$BeginVector" << endl;
  for (int i = 0; i < p.v.rows(); ++i)
    datafile << p.v(i) << endl;
  datafile << "$EndVector" << endl;

  datafile.close();
}