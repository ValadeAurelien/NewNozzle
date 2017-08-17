#include "flow.cu"
#define CUDA_CALLABLE_MEMBER_OVERWRITE
#define CUDA_CALLABLE_MEMBER
#include "../Solver/solver.cu"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

__global__
void init(dev_meshgrid_t* m, data_t zeta, data_t R, data_t M)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;

  if (i>m->size_i-1 || j>m->size_j-1) return;

  m->dat(i, j).vr = 0;
  m->dat(i, j).vz = 0;
  m->dat(i, j).rho = 10-9*(data_t) i/m->size_i; //-9*sqrtf( powf( (data_t)i/m->size_i, 2) + powf( (data_t)j/m->size_j, 2));
  m->dat(i, j).T = 1;
  m->dat(i, j).P = zeta * R / M * m->dat(i, j).rho * m->dat(i, j).T; 
  m->dat(i, j).is_wall = false;
}

meshgrid_t test(meshgrid_t m)
{
  printf("lcl %p %p %p\n", &m, m.d_data, m.h_data); 
  return m;
}

int main(int argc, char **argv)
{
  data_t t0 = atof(argv[1]),
         v0 = atof(argv[2]),
         D = atof(argv[3]),
         rho0 = atof(argv[4]),
         P0 = atof(argv[5]), 
         T0 = atof(argv[6]),
         eta = atof(argv[7]),
         lambda = atof(argv[8]),
         R = atof(argv[9]),
         M = atof(argv[10]),
         C = atof(argv[11]), 
         dt = atof(argv[12]), 
         t_tot = atof(argv[13]);
  int size_i = atoi(argv[14]),
      size_j = atoi(argv[15]);
  dim3 gridsize(64,64), 
       blocksize(32,32);
  
  Flow_t Flow(gridsize,blocksize, 
              t0, v0, D, rho0, P0, T0, 
              eta, lambda, R, M, C); 

  meshgrid_t m1(size_i, size_j, gridsize, blocksize),
             m2(size_i, size_j, gridsize, blocksize);
  m1.reserve_host_data();
  m2.reserve_host_data();

  init<<<gridsize, blocksize>>>(m1.d_m, Flow.get_zeta(), R, M);
  m1.data_to_host();
//  cout << m1 << endl;

 
  RK45Solver_t<Flow_t, meshgrid_t, data_t> Solver(Flow, 0, 1);

//  Solver(dt, 1, m1, m2);
//  Solver(dt, 1, m2, m1);
  for (data_t t=0; t<t_tot; t+=2*dt)
  {
    m1.data_to_host();
    for (int j=0; j<size_j; j++)
    {
      for (int i=0; i<size_i; i++)
        cout << t << " " << i << " " << j << " " 
             << m1.cat(i, j).vr << " " << m1.cat(i, j).vz << " " << m1.cat(i, j).rho << " " << m1.cat(i, j).T << " " << m1.cat(i, j).P << endl;
      cout << endl;
    }
    cout << endl ;
    Solver(dt, 1, m1, m2);
    Solver(dt, 1, m2, m1);
  }

  return 0;
};

