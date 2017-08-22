#include "meshgrid.cu"
#include <iostream>
using namespace std;

class Flow_t
{
  public : 
    Flow_t() {}
    Flow_t(dim3 _gridsize, dim3 _blocksize, 
           data_t _t0, data_t _v0, data_t _D, data_t _rho0, data_t _P0, data_t _T0, 
           data_t _eta, data_t _lambda, data_t _R, data_t _M, data_t _C); 
    meshgrid_t operator()(const meshgrid_t& m);

		__host__ __device__
    inline data_t get_R() const { return R; }
		__host__ __device__
    inline data_t get_M() const { return M; }
		__host__ __device__
    inline data_t get_C() const { return C; }
		__host__ __device__
    inline data_t get_eta() const { return eta; }
		__host__ __device__
    inline data_t get_lambda() const { return lambda; }
		__host__ __device__
    inline data_t get_alpha() const { return alpha; }
		__host__ __device__
    inline data_t get_beta() const { return beta; }
		__host__ __device__
    inline data_t get_gamma() const { return gamma; } 
		__host__ __device__
    inline data_t get_delta() const { return delta; } 
		__host__ __device__
    inline data_t get_epsilon() const { return epsilon; } 
		__host__ __device__
    inline data_t get_zeta() const { return zeta; }

  private:
    data_t t0, v0, D, rho0, P0, T0, 
           eta, lambda,
           R, M, C, 
           alpha, beta, gamma, 
           delta, epsilon, zeta, theta;
    dim3 gridsize, blocksize;
    meshgrid_t out;
};

//----- dr -----
__device__
inline data_t dr_vr(const dev_meshgrid_t *in, int i, int j) 
{
  data_t tv, bv;
  if (in->cdat(i, j-1).is_wall) bv = 0;
  else bv = in->cdat(i, j-1).vr;
  if (in->cdat(i, j+1).is_wall) tv = 0;
  else tv = in->cdat(i, j+1).vr;
  return in->size_j*(tv-bv)/2;
}

__device__
inline data_t dr_vz(const dev_meshgrid_t *in, int i, int j) 
{
  data_t tv, bv;
  if (in->cdat(i, j-1).is_wall) bv = 0;
  else bv = in->cdat(i, j-1).vz;
  if (in->cdat(i, j+1).is_wall) tv = 0;
  else tv = in->cdat(i, j+1).vz;
  return in->size_j*(tv-bv)/2;
}

__device__
inline data_t dr_rho(const dev_meshgrid_t *in, int i, int j) 
{
  data_t tv, bv;
  if (in->cdat(i, j-1).is_wall) return 0;
  else bv = in->cdat(i, j-1).rho;
  if (in->cdat(i, j+1).is_wall) return 0;
  else tv = in->cdat(i, j+1).rho;
  return in->size_j*(tv-bv)/2;
}

__device__
inline data_t dr_P(const dev_meshgrid_t *in, int i, int j) 
{
  data_t tv, bv;
  if (in->cdat(i, j-1).is_wall) return 0;
  else bv = in->cdat(i, j-1).P;
  if (in->cdat(i, j+1).is_wall) return 0;
  else tv = in->cdat(i, j+1).P;
  return in->size_j*(tv-bv)/2;
}

__device__
inline data_t dr_T(const dev_meshgrid_t *in, int i, int j) 
{
  data_t tv, bv;
  if (in->cdat(i, j-1).is_wall) return 0;
  else bv = in->cdat(i, j-1).T;
  if (in->cdat(i, j+1).is_wall) return 0;
  else tv = in->cdat(i, j+1).T;
  return in->size_j*(tv-bv)/2;
}

//----- dz -----

__device__
inline data_t dz_vr(const dev_meshgrid_t *in, int i, int j) 
{
  if (i==in->size_i-1) return dz_vr(in, i-1, j);
  data_t lv, rv;
  if (in->cdat(i-1, j).is_wall) lv = 0;
  else lv = in->cdat(i-1, j).vr;
  if (in->cdat(i+1, j).is_wall) rv = 0;
  else rv = in->cdat(i+1, j).vr;
  return in->size_i*(rv-lv)/2;
}
    
__device__
inline data_t dz_vz(const dev_meshgrid_t *in, int i, int j) 
{
  if (i==in->size_i-1) return dz_vz(in, i-1, j);
  data_t lv, rv;
  if (in->cdat(i-1, j).is_wall) lv = 0;
  else lv = in->cdat(i-1, j).vz;
  if (in->cdat(i+1, j).is_wall) rv = 0;
  else rv = in->cdat(i+1, j).vz;
  return in->size_i*(rv-lv)/2;
}
    
__device__
inline data_t dz_rho(const dev_meshgrid_t *in, int i, int j) 
{
  if (i==in->size_i-1) return dz_rho(in, i-1, j);
  data_t lv, rv;
  if (in->cdat(i-1, j).is_wall) return 0; 
  else lv = in->cdat(i-1, j).rho;
  if (in->cdat(i+1, j).is_wall) return 0; 
  else rv = in->cdat(i+1, j).rho;
  return in->size_i*(rv-lv)/2;
}
    
__device__
inline data_t dz_P(const dev_meshgrid_t *in, int i, int j) 
{
  if (i==in->size_i-1) return dz_P(in, i-1, j);
  data_t lv, rv;
  if (in->cdat(i-1, j).is_wall) return 0;
  else lv = in->cdat(i-1, j).P;
  if (in->cdat(i+1, j).is_wall) return 0;
  else rv = in->cdat(i+1, j).P;
  return in->size_i*(rv-lv)/2;
}
    
__device__
inline data_t dz_T(const dev_meshgrid_t *in, int i, int j) 
{
  if (i==in->size_i-1) return dz_T(in, i-1, j);
  data_t lv, rv;
  if (in->cdat(i-1, j).is_wall) return 0;
  else lv = in->cdat(i-1, j).vz;
  if (in->cdat(i+1, j).is_wall) return 0;
  else rv = in->cdat(i+1, j).vz;
  return in->size_i*(rv-lv)/2;
}
   
//----- d2r -----

__device__
inline data_t d2r_vr(const dev_meshgrid_t *in, int i, int j) 
{
  data_t tv, bv;
  if (in->cdat(i, j-1).is_wall) bv = 0;
  else bv = in->cdat(i, j-1).vr;
  if (in->cdat(i, j+1).is_wall) tv = 0;
  else tv = in->cdat(i, j+1).vr;
  return in->size_j*(tv+bv-2*in->cdat(i,j).vr);
}

__device__
inline data_t d2r_vz(const dev_meshgrid_t *in, int i, int j) 
{
  data_t tv, bv;
  if (in->cdat(i, j-1).is_wall) bv = 0;
  else bv = in->cdat(i, j-1).vz;
  if (in->cdat(i, j+1).is_wall) tv = 0;
  else tv = in->cdat(i, j+1).vz;
  return in->size_j*(tv+bv-2*in->cdat(i,j).vz);
}

__device__
inline data_t d2r_T(const dev_meshgrid_t *in, int i, int j) 
{
  data_t tv, bv;
  if (in->cdat(i, j-1).is_wall) bv = in->cdat(i,j).T;
  else bv = in->cdat(i, j-1).T;
  if (in->cdat(i, j+1).is_wall) tv = in->cdat(i,j).T;
  else tv = in->cdat(i, j+1).T;
  return in->size_j*(tv+bv-2*in->cdat(i,j).T);
}

__device__
inline data_t d2r_P(const dev_meshgrid_t *in, int i, int j) 
{
  data_t tv, bv;
  if (in->cdat(i, j-1).is_wall) bv = in->cdat(i,j).P;
  else bv = in->cdat(i, j-1).P;
  if (in->cdat(i, j+1).is_wall) tv = in->cdat(i,j).P;
  else tv = in->cdat(i, j+1).P;
  return in->size_j*(tv+bv-2*in->cdat(i,j).P);
}

__device__
inline data_t d2z_vr(const dev_meshgrid_t *in, int i, int j) 
{
  data_t rv, lv;
  if (in->cdat(i-1, j).is_wall) lv = 0;
  else lv = in->cdat(i-1, j).vr;
  if (in->cdat(i+1, j).is_wall) rv = 0;
  else rv = in->cdat(i+1, j).vr;
  return in->size_j*(rv+lv-2*in->cdat(i,j).vr);
}

__device__
inline data_t d2z_vz(const dev_meshgrid_t *in, int i, int j) 
{
  data_t rv, lv;
  if (in->cdat(i-1, j).is_wall) lv = 0;
  else lv = in->cdat(i-1, j).vz;
  if (in->cdat(i+1, j).is_wall) rv = 0;
  else rv = in->cdat(i+1, j).vz;
  return in->size_j*(rv+lv-2*in->cdat(i,j).vz);
}

__device__
inline data_t d2z_T(const dev_meshgrid_t *in, int i, int j) 
{
  data_t rv, lv;
  if (in->cdat(i-1, j).is_wall) lv = in->cdat(i,j).T;
  else lv = in->cdat(i-1, j).T;
  if (in->cdat(i+1, j).is_wall) rv = in->cdat(i,j).T;
  else rv = in->cdat(i+1, j).T;
  return in->size_j*(rv+lv-2*in->cdat(i,j).T);
}

__device__
inline data_t d2z_P(const dev_meshgrid_t *in, int i, int j) 
{
  data_t rv, lv;
  if (in->cdat(i-1, j).is_wall) lv = in->cdat(i,j).P;
  else lv = in->cdat(i-1, j).P;
  if (in->cdat(i+1, j).is_wall) rv = in->cdat(i,j).P;
  else rv = in->cdat(i+1, j).P;
  return in->size_j*(rv+lv-2*in->cdat(i,j).P);
}

__device__
inline data_t div_v(const dev_meshgrid_t *in, int i, int j)
{
  data_t val=dr_vr(in, i, j) + dz_vz(in, i, j);
  if (j>0) val += in->size_j*in->cdat(i, j).vr/j;
  return val;
}

__device__
inline data_t vgrad_vr(const dev_meshgrid_t *in, int i, int j) 
{
  return in->cdat(i,j).vr * dr_vr(in, i, j) 
       + in->cdat(i,j).vz * dz_vr(in, i, j) ;
}

__device__
inline data_t vgrad_vz(const dev_meshgrid_t *in, int i, int j) 
{
  return in->cdat(i,j).vr * dr_vz(in, i, j) 
       + in->cdat(i,j).vz * dz_vz(in, i, j) ;
}

__device__
inline data_t vgrad_rho(const dev_meshgrid_t *in, int i, int j) 
{
  return in->cdat(i,j).vr * dr_rho(in, i, j) 
       + in->cdat(i,j).vz * dz_rho(in, i, j) ;
}
__device__
inline data_t vgrad_P(const dev_meshgrid_t *in, int i, int j) 
{
  return in->cdat(i,j).vr * dr_P(in, i, j) 
       + in->cdat(i,j).vz * dz_P(in, i, j) ;
}

__device__
inline data_t vgrad_T(const dev_meshgrid_t *in, int i, int j) 
{
  return in->cdat(i,j).vr * dr_T(in, i, j) 
       + in->cdat(i,j).vz * dz_T(in, i, j) ;
}

__device__
inline data_t lapsca_P(const dev_meshgrid_t *in, int i, int j) 
{
  data_t val = d2r_P(in, i, j) + d2z_P(in, i, j);
  if (j>0) val += (data_t) in->size_j/j * dr_P(in, i, j);
  return val;
}

__device__
inline data_t lapsca_T(const dev_meshgrid_t *in, int i, int j) 
{
  data_t val = d2r_T(in, i, j) + d2z_T(in, i, j);
  if (j>0) val += (data_t) in->size_j/j * dr_T(in, i, j);
  return val;
}

__device__
inline data_t lapvec_vr(const dev_meshgrid_t *in, int i, int j) 
{
  data_t val = d2r_vr(in, i, j) + d2z_vr(in, i, j);
  if (j>0) val += (data_t) in->size_j/j * dr_vr(in, i, j) - 
                  (data_t) (in->size_j*in->size_j)/(j*j) * in->cdat(i,j).vr; //vr(r=0, z) = 0 !!!!!!
  return val;
}

__device__
inline data_t lapvec_vz(const dev_meshgrid_t *in, int i, int j) 
{
  data_t val = d2r_vz(in, i, j) + d2z_vz(in, i, j);
  if (j>0) val += (data_t) in->size_j/j * dr_vz(in, i, j);
  return val;
}

//----- update -----
__global__ 
void one_step(dev_meshgrid_t *out, const dev_meshgrid_t *in, 
              const data_t alpha, const data_t beta, const data_t gamma, 
              const data_t delta, const data_t epsilon, const data_t zeta, const data_t theta,
              const data_t eta, const data_t lambda, 
              const data_t R, const data_t M, const data_t C)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;

  if (i>in->size_i-1 || j>in->size_j-1) return;
  else {
    cell_t& out_c = out->dat(i, j);
    const cell_t& in_c = in->cdat(i, j);
    if (i==0) {
      out_c.vr = 0;
      out_c.vz = 0;
      out_c.rho = 0;
      out_c.T = 0;
      out_c.P = 0;
    }
    else {
      if (in_c.is_wall) return;
      else {
        data_t rmc = R/(M*C),
               inv_rmc = 1./(1-rmc);

        out_c.vr = alpha * ( in_c.vr * div_v(in, i, j) - vgrad_vr(in, i, j) )
                 + beta * eta / in_c.rho * lapvec_vr(in, i, j)
                 - gamma / in_c.rho * dr_P(in, i, j);

        out_c.vz = alpha * ( in_c.vz * div_v(in, i, j) - vgrad_vz(in, i, j) )
                 + beta * eta / in_c.rho * lapvec_vz(in, i, j)
                 - gamma / in_c.rho * dz_P(in, i, j);

        out_c.rho = - alpha * ( in_c.rho * div_v(in, i, j) + vgrad_rho(in, i, j) );
        
        out_c.T = 0;
//        out_c.T = inv_rmc * rmc * out_c.rho
//                + delta * inv_rmc * eta * ( lapvec_vr(in, i, j) * in_c.vr + lapvec_vz(in, i, j) * in_c.vz )
//                + epsilon * inv_rmc * lambda * lapsca_T(in, i, j)
//                - zeta * inv_rmc * vgrad_T(in, i, j);
//
        out_c.P = theta * R / M * ( out_c.rho * in_c.T + out_c.T * in_c.rho ) ;
        
//        if (j==25 && ( i==59 || i==60 || i==61 || i ==62 ) )
//          printf("%d %f %f %f %f %f\n", i, 
//                 out_c.vz, - alpha * vgrad_vz(in, i, j), vgrad_vz(in, i, j), in_c.vz, dz_vz(in, i, j));
      }
    }
  }
};



//----- FLOW ------
Flow_t::Flow_t(dim3 _gridsize, dim3 _blocksize, 
       data_t _t0, data_t _v0, data_t _D, data_t _rho0, data_t _P0, data_t _T0, 
       data_t _eta, data_t _lambda, data_t _R, data_t _M, data_t _C) :
  gridsize(_gridsize), blocksize(_blocksize), 
  t0(_t0), v0(_v0), D(_D), rho0(_rho0), P0(_P0), T0(_T0), 
  eta(_eta), lambda(_lambda), R(_R), M(_M), C(_C) 
{
  alpha = t0*v0/D;
  beta = t0/(rho0*D);
  gamma = t0*P0/(v0*rho0*D);
  delta = t0*v0*v0/(T0*D);
  epsilon = t0/(D*D);
  zeta = v0*T0/D;
  theta = rho0*T0/P0;
}

meshgrid_t Flow_t::operator()(const meshgrid_t& m)
{
  out = m;
  one_step<<<gridsize, blocksize>>>(out.d_m, m.d_m, 
                                    alpha, beta, gamma, 
                                    delta, epsilon, zeta, theta,
                                    eta, lambda, 
                                    R, M, C);
  return out;
}
  
  

  
//__global__
//void test(const dev_meshgrid_t * in)
//{
//  int i = blockIdx.x*blockDim.x+threadIdx.x;
//  int j = blockIdx.y*blockDim.y+threadIdx.y;
//
//  if (i>in->size_i-1 || j>in->size_j-1) return;
//  printf("%d %d %f %f %f %f %f\n", i, j, in->cdat(i, j).vr, in->cdat(i, j).vz, in->cdat(i, j).rho, in->cdat(i, j).T, in->cdat(i, j).P);
//}
//
//__global__
//void testbis(const cell_t * in, int size_i, int size_j)
//{
//  int i = blockIdx.x*blockDim.x+threadIdx.x;
//  int j = blockIdx.y*blockDim.y+threadIdx.y;
//
//  if (i>size_i-1 || j>size_j-1) return;
//  printf("%d %d %f %f %f %f %f\n", i, j, in[j*size_i+i].vr, in[j*size_i+i].vz, in[j*size_i+i].rho, in[j*size_i+i].T, in[j*size_i+i].P);
//}

//      if (i==1 && j==25)
//      {
//        printf("%d %d (i+1,j) %f %f %f %f %f\n", i, j, 
//                in->cdat(i+1,j).vr, in->cdat(i+1,j).vz, in->cdat(i+1,j).rho, in->cdat(i+1,j).T, in->cdat(i+1,j).P);
//        printf("%d %d (i-1,j) %f %f %f %f %f\n", i, j, 
//                in->cdat(i-1,j).vr, in->cdat(i-1,j).vz, in->cdat(i-1,j).rho, in->cdat(i-1,j).T, in->cdat(i-1,j).P);
//        printf("%d %d (i,j) %f %f %f %f %f\n", i, j, 
//                in->cdat(i,j).vr, in->cdat(i,j).vz, in->cdat(i,j).rho, in->cdat(i,j).T, in->cdat(i,j).P);
//        printf("%d %d (i,j+1) %f %f %f %f %f\n", i, j, 
//                in->cdat(i,j+1).vr, in->cdat(i,j+1).vz, in->cdat(i,j+1).rho, in->cdat(i,j+1).T, in->cdat(i,j+1).P);
//        printf("%d %d (i,j-1) %f %f %f %f %f\n", i, j, 
//                in->cdat(i,j-1).vr, in->cdat(i,j-1).vz, in->cdat(i,j-1).rho, in->cdat(i,j-1).T, in->cdat(i,j-1).P);
//        printf("%d %d d(i+1,j) %f %f %f %f %f\n", i, j, 
//               dz_vr(in,i+1,j), dz_vz(in,i+1,j), dz_rho(in,i+1,j), dz_T(in,i+1,j), dz_P(in,i+1,j));
//        printf("%d %d d(i-1,j) %f %f %f %f %f\n", i, j, 
//               dz_vr(in,i-1,j), dz_vz(in,i-1,j), dz_rho(in,i-1,j), dz_T(in,i-1,j), dz_P(in,i-1,j));
//        printf("%d %d d(i,j) %f %f %f %f %f\n", i, j, 
//               dz_vr(in,i,j), dz_vz(in,i,j), dz_rho(in,i,j), dz_T(in,i,j), dz_P(in,i,j));
//        printf("%d %d d(i,j+1) %f %f %f %f %f\n", i, j, 
//               dz_vr(in,i,j+1), dz_vz(in,i,j+1), dz_rho(in,i,j+1), dz_T(in,i,j+1), dz_P(in,i,j+1));
//        printf("%d %d d(i,j-1) %f %f %f %f %f\n", i, j, 
//               dz_vr(in,i,j-1), dz_vz(in,i,j-1), dz_rho(in,i,j-1), dz_T(in,i,j-1), dz_P(in,i,j-1));
//        printf("#\n");
    //    printf("%d %d calc %f %f %f %f %f out %f %f %f %f %f\n", i, j, 
    //           dz_vr(in, i, j), dz_vz(in, i, j), dz_rho(in, i, j), dz_T(in, i, j), dz_P(in, i, j),
    //           out_c.vr, out_c.vz, out_c.rho, out_c.T, out_c.P);
//      }
