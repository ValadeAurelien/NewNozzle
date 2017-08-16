#include "meshgrid.cu"

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

		__host__ __device__
    inline const meshgrid_t& get_in() const { return in; }
		__host__ __device__
    inline meshgrid_t& get_out() { return out; }

//    //----- dr -----
//    __host__ __device__
//    data_t dr_vr(int i, int j) const;
//    __host__ __device__
//    data_t dr_vz(int i, int j) const;
//    __host__ __device__
//    data_t dr_rho(int i, int j) const;
//    __host__ __device__
//    data_t dr_P(int i, int j) const;
//    __host__ __device__
//    data_t dr_T(int i, int j) const;
//    //----- dz -----
//    __host__ __device__
//    data_t dz_vr(int i, int j) const;
//    __host__ __device__
//    data_t dz_vz(int i, int j) const;
//    __host__ __device__
//    data_t dz_rho(int i, int j) const;
//    __host__ __device__
//    data_t dz_P(int i, int j) const;
//    __host__ __device__
//    data_t dz_T(int i, int j) const;
//       
//    //----- d2r -----
//    __host__ __device__
//    data_t d2r_vr(int i, int j) const;
//    __host__ __device__
//    data_t d2r_vz(int i, int j) const;
//    __host__ __device__
//    data_t d2r_T(int i, int j) const;
//    __host__ __device__
//    data_t d2r_P(int i, int j) const;
//    __host__ __device__
//    data_t d2z_vr(int i, int j) const;
//    __host__ __device__
//    data_t d2z_vz(int i, int j) const;
//    __host__ __device__
//    data_t d2z_T(int i, int j) const;
//    __host__ __device__
//    data_t d2z_P(int i, int j) const;
//
//    __host__ __device__
//    data_t vgrad_vr(int i, int j) const;
//    __host__ __device__
//    data_t vgrad_vz(int i, int j) const;
//    __host__ __device__
//    data_t vgrad_P(int i, int j) const;
//    __host__ __device__
//    data_t vgrad_T(int i, int j) const;
//    __host__ __device__
//    data_t lapsca_P(int i, int j) const;
//    __host__ __device__
//    data_t lapsca_T(int i, int j) const;
//    __host__ __device__
//    data_t lapvec_vr(int i, int j) const;
//    __host__ __device__
//    data_t lapvec_vz(int i, int j) const;

  private:
    data_t t0, v0, D, rho0, P0, T0, 
           eta, lambda,
           R, M, C, 
           alpha, beta, gamma, 
           delta, epsilon, zeta;
    meshgrid_t in, out;
    dim3 gridsize, blocksize;
};

//----- dr -----
__device__
data_t dr_vr(dev_meshgrid_t *in, int i, int j) 
{
  data_t tv, bv;
  if (j==0) return 0;
  else
  {
    if (in->cdat(i, j-1).is_wall) bv = 0;
    else bv = in->cdat(i, j-1).vr;
  }
  if (j==in->size_j-1) tv = 0;
  else
  {
    if (in->cdat(i, j+1).is_wall) tv = 0;
    else tv = in->cdat(i, j+1).vr;
  }
  return in->size_j*(tv-bv)/2;
}

__device__
data_t dr_vz(dev_meshgrid_t *in, int i, int j) 
{
  data_t tv, bv;
  if (j==0) return 0;
  else
  {
    if (in->cdat(i, j-1).is_wall) bv = 0;
    else bv = in->cdat(i, j-1).vz;
  }
  if (j==in->size_j-1) tv = 0;
  else
  {
    if (in->cdat(i, j+1).is_wall) tv = 0;
    else tv = in->cdat(i, j+1).vz;
  }
  return in->size_j*(tv-bv)/2;
}

__device__
data_t dr_rho(dev_meshgrid_t *in, int i, int j) 
{
  data_t tv, bv;
  if (j==0) return 0;
  else
  {
    if (in->cdat(i, j-1).is_wall) return 0;
    else bv = in->cdat(i, j-1).rho;
  }
  if (j==in->size_j-1) tv = 0;
  else
  {
    if (in->cdat(i, j+1).is_wall) return 0;
    else tv = in->cdat(i, j+1).rho;
  }
  return in->size_j*(tv-bv)/2;
}

__device__
data_t dr_P(dev_meshgrid_t *in, int i, int j) 
{
  data_t tv, bv;
  if (j==0) return 0;
  else
  {
    if (in->cdat(i, j-1).is_wall) return 0;
    else bv = in->cdat(i, j-1).P;
  }
  if (j==in->size_j-1) tv = 0;
  else
  {
    if (in->cdat(i, j+1).is_wall) return 0;
    else tv = in->cdat(i, j+1).P;
  }
  return in->size_j*(tv-bv)/2;
}

__device__
data_t dr_T(dev_meshgrid_t *in, int i, int j) 
{
  data_t tv, bv;
  if (j==0) return 0;
  else
  {
    if (in->cdat(i, j-1).is_wall) return 0;
    else bv = in->cdat(i, j-1).T;
  }
  if (j==in->size_j-1) tv = 0;
  else
  {
    if (in->cdat(i, j+1).is_wall) return 0;
    else tv = in->cdat(i, j+1).T;
  }
  return in->size_j*(tv-bv)/2;
}

//----- dz -----

__device__
data_t dz_vr(dev_meshgrid_t *in, int i, int j) 
{
  data_t lv, rv;
  if (i==0) lv = 0;
  else
  {
    if (in->cdat(i-1, j).is_wall) lv = 0;
    else lv = in->cdat(i-1, j).vr;
  }
  if (i==in->size_i-1) rv = 0;
  else
  {
    if (in->cdat(i+1, j).is_wall) rv = 0;
    else rv = in->cdat(i+1, j).vr;
  }
  return in->size_i*(rv-lv)/2;
}
    
__device__
data_t dz_vz(dev_meshgrid_t *in, int i, int j) 
{
  data_t lv, rv;
  if (i==0) lv = 0;
  else
  {
    if (in->cdat(i-1, j).is_wall) lv = 0;
    else lv = in->cdat(i-1, j).vz;
  }
  if (i==in->size_i-1) rv = 0;
  else
  {
    if (in->cdat(i+1, j).is_wall) rv = 0;
    else rv = in->cdat(i+1, j).vz;
  }
  return in->size_i*(rv-lv)/2;
}
    
__device__
data_t dz_rho(dev_meshgrid_t *in, int i, int j) 
{
  data_t lv, rv;
  if (i==0) lv = 0;
  else
  {
    if (in->cdat(i-1, j).is_wall) return 0; 
    else lv = in->cdat(i-1, j).rho;
  }
  if (i==in->size_i-1) rv = 0;
  else
  {
    if (in->cdat(i+1, j).is_wall) return 0; 
    else rv = in->cdat(i+1, j).vz;
  }
  return in->size_i*(rv-lv)/2;
}
    
__device__
data_t dz_P(dev_meshgrid_t *in, int i, int j) 
{
  data_t lv, rv;
  if (i==0) lv = 0;
  else
  {
    if (in->cdat(i-1, j).is_wall) return 0;
    else lv = in->cdat(i-1, j).vz;
  }
  if (i==in->size_i-1) rv = 0;
  else
  {
    if (in->cdat(i+1, j).is_wall) return 0;
    else rv = in->cdat(i+1, j).vz;
  }
  return in->size_i*(rv-lv)/2;
}
    
__device__
data_t dz_T(dev_meshgrid_t *in, int i, int j) 
{
  data_t lv, rv;
  if (i==0) lv = 0;
  else
  {
    if (in->cdat(i-1, j).is_wall) return 0;
    else lv = in->cdat(i-1, j).vz;
  }
  if (i==in->size_i-1) rv = 0;
  else
  {
    if (in->cdat(i+1, j).is_wall) return 0;
    else rv = in->cdat(i+1, j).vz;
  }
  return in->size_i*(rv-lv)/2;
}
   
//----- d2r -----

__device__
data_t d2r_vr(dev_meshgrid_t *in, int i, int j) 
{
  data_t tv, bv;
  if (j==0) bv=in->cdat(i,j).vr;
  else
  {
    if (in->cdat(i, j-1).is_wall) bv = 0;
    else bv = in->cdat(i, j-1).vr;
  }
  if (j==in->size_j-1) tv = in->cdat(i,j).vr;
  else
  {
    if (in->cdat(i, j+1).is_wall) tv = 0;
    else tv = in->cdat(i, j+1).vr;
  }
  return in->size_j*(tv+bv-2*in->cdat(i,j).vr);
}

__device__
data_t d2r_vz(dev_meshgrid_t *in, int i, int j) 
{
  data_t tv, bv;
  if (j==0) bv=in->cdat(i,j).vz;
  else
  {
    if (in->cdat(i, j-1).is_wall) bv = 0;
    else bv = in->cdat(i, j-1).vz;
  }
  if (j==in->size_j-1) tv = in->cdat(i,j).vz;
  else
  {
    if (in->cdat(i, j+1).is_wall) tv = 0;
    else tv = in->cdat(i, j+1).vz;
  }
  return in->size_j*(tv+bv-2*in->cdat(i,j).vz);
}

__device__
data_t d2r_T(dev_meshgrid_t *in, int i, int j) 
{
  data_t tv, bv;
  if (j==0) bv=in->cdat(i,j).T;
  else
  {
    if (in->cdat(i, j-1).is_wall) bv = in->cdat(i,j).T;
    else bv = in->cdat(i, j-1).vz;
  }
  if (j==in->size_j-1) tv = in->cdat(i,j).T;
  else
  {
    if (in->cdat(i, j+1).is_wall) tv = in->cdat(i,j).T;
    else tv = in->cdat(i, j+1).vz;
  }
  return in->size_j*(tv+bv-2*in->cdat(i,j).T);
}

__device__
data_t d2r_P(dev_meshgrid_t *in, int i, int j) 
{
  data_t tv, bv;
  if (j==0) bv=in->cdat(i,j).P;
  else
  {
    if (in->cdat(i, j-1).is_wall) bv = in->cdat(i,j).P;
    else bv = in->cdat(i, j-1).P;
  }
  if (j==in->size_j-1) tv = in->cdat(i,j).P;
  else
  {
    if (in->cdat(i, j+1).is_wall) tv = in->cdat(i,j).P;
    else tv = in->cdat(i, j+1).vz;
  }
  return in->size_j*(tv+bv-2*in->cdat(i,j).P);
}

__device__
data_t d2z_vr(dev_meshgrid_t *in, int i, int j) 
{
  data_t rv, lv;
  if (i==0) lv=in->cdat(i,j).vr;
  else
  {
    if (in->cdat(i-1, j).is_wall) lv = 0;
    else lv = in->cdat(i-1, j).vr;
  }
  if (i==in->size_i-1) rv = in->cdat(i,j).vr;
  else
  {
    if (in->cdat(i+1, j).is_wall) rv = 0;
    else rv = in->cdat(i+1, j).vr;
  }
  return in->size_j*(rv+lv-2*in->cdat(i,j).vr);
}

__device__
data_t d2z_vz(dev_meshgrid_t *in, int i, int j) 
{
  data_t rv, lv;
  if (i==0) lv=in->cdat(i,j).vz;
  else
  {
    if (in->cdat(i-1, j).is_wall) lv = 0;
    else lv = in->cdat(i-1, j).vz;
  }
  if (i==in->size_i-1) rv = in->cdat(i,j).vz;
  else
  {
    if (in->cdat(i+1, j).is_wall) rv = 0;
    else rv = in->cdat(i+1, j).vz;
  }
  return in->size_j*(rv+lv-2*in->cdat(i,j).vz);
}

__device__
data_t d2z_T(dev_meshgrid_t *in, int i, int j) 
{
  data_t rv, lv;
  if (i==0) lv=in->cdat(i,j).T;
  else
  {
    if (in->cdat(i-1, j).is_wall) lv = 0;
    else lv = in->cdat(i-1, j).T;
  }
  if (i==in->size_i-1) rv = in->cdat(i,j).T;
  else
  {
    if (in->cdat(i+1, j).is_wall) rv = 0;
    else rv = in->cdat(i+1, j).T;
  }
  return in->size_j*(rv+lv-2*in->cdat(i,j).T);
}

__device__
data_t d2z_P(dev_meshgrid_t *in, int i, int j) 
{
  data_t rv, lv;
  if (i==0) lv=in->cdat(i,j).vz;
  else
  {
    if (in->cdat(i-1, j).is_wall) lv = 0;
    else lv = in->cdat(i-1, j).P;
  }
  if (i==in->size_i-1) rv = in->cdat(i,j).P;
  else
  {
    if (in->cdat(i+1, j).is_wall) rv = 0;
    else rv = in->cdat(i+1, j).P;
  }
  return in->size_j*(rv+lv-2*in->cdat(i,j).P);
}


__device__
data_t vgrad_vr(dev_meshgrid_t *in, int i, int j) 
{
  return in->cdat(i,j).vr * dr_vr(in, i, j) 
       + in->cdat(i,j).vz * dz_vr(in, i, j) ;
}

__device__
data_t vgrad_vz(dev_meshgrid_t *in, int i, int j) 
{
  return in->cdat(i,j).vr * dr_vz(in, i, j) 
       + in->cdat(i,j).vz * dz_vz(in, i, j) ;
}

__device__
data_t vgrad_P(dev_meshgrid_t *in, int i, int j) 
{
  return in->cdat(i,j).vr * dr_P(in, i, j) 
       + in->cdat(i,j).vz * dz_P(in, i, j) ;
}

__device__
data_t vgrad_T(dev_meshgrid_t *in, int i, int j) 
{
  return in->cdat(i,j).vr * dr_T(in, i, j) 
       + in->cdat(i,j).vz * dz_T(in, i, j) ;
}

__device__
data_t lapsca_P(dev_meshgrid_t *in, int i, int j) 
{
  data_t val = d2r_P(in, i, j) + d2z_P(in, i, j);
  if (j>0) val += (data_t) in->size_j/j * dr_P(in, i, j);
  return val;
}

__device__
data_t lapsca_T(dev_meshgrid_t *in, int i, int j) 
{
  data_t val = d2r_T(in, i, j) + d2z_T(in, i, j);
  if (j>0) val += (data_t) in->size_j/j * dr_T(in, i, j);
  return val;
}

__device__
data_t lapvec_vr(dev_meshgrid_t *in, int i, int j) 
{
  data_t val = d2r_vr(in, i, j) + d2z_vr(in, i, j);
  if (j>0) val += (data_t) in->size_j/j * dr_vr(in, i, j) - 
                  (data_t) (in->size_j*in->size_j)/(j*j) * in->cdat(i,j).vr; //vr(r=0, z) = 0 !!!!!!
  return val;
}

__device__
data_t lapvec_vz(dev_meshgrid_t *in, int i, int j) 
{
  data_t val = d2r_vz(in, i, j) + d2z_vz(in, i, j);
  if (j>0) val += (data_t) in->size_j/j * dr_vz(in, i, j);
  return val;
}

//----- update -----
__global__ 
void one_step(dev_meshgrid_t *out, dev_meshgrid_t *in, 
              data_t alpha, data_t beta, data_t gamma, 
              data_t delta, data_t epsilon, data_t zeta,
              data_t eta, data_t lambda, 
              data_t R, data_t M, data_t C)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;

  if (i>in->size_i || j>in->size_j) return;

  cell_t& out_c = in->dat(i, j);
  const cell_t& in_c = in->cdat(i, j);

  data_t rmc = R/(M*C),
         inv_rmc = 1./(1-rmc);

  out_c.vr = alpha * vgrad_vr(in, i, j)
           + beta * eta / in_c.rho * lapvec_vr(in, i, j)
           - gamma / in_c.rho * dr_P(in, i, j);

  out_c.vz = alpha * vgrad_vz(in, i, j)
           + beta * eta / in_c.rho * lapvec_vz(in, i, j)
           - gamma / in_c.rho * dz_P(in, i, j);

  out_c.rho = alpha * ( in_c.rho * in_c.vr * ( (data_t) in->size_j/j )
            + in_c.vr * dr_rho(in, i, j) + in_c.rho * dr_vr(in, i, j)
            + in_c.vz * dz_rho(in, i, j) + in_c.rho * dz_vz(in, i, j) );
  
  out_c.T = inv_rmc * rmc * out_c.rho 
          + delta * inv_rmc * eta * ( lapvec_vr(in, i, j) * in_c.vr + lapvec_vz(in, i, j) * in_c.vz )
          + epsilon * inv_rmc * lambda * lapsca_T(in, i, j); 

  out_c.P = zeta * R / M * ( out_c.rho * in_c.T + out_c.T * in_c.rho ) ;
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
  zeta = rho0*T0/P0;
}

meshgrid_t Flow_t::operator()(const meshgrid_t& m)
{
  out.from_meshgrid_prop(m);
  in = m;
  one_step<<<gridsize, blocksize>>>(out.d_m, in.d_m, 
                                    alpha, beta, gamma, 
                                    delta, epsilon, zeta,
                                    eta, lambda, 
                                    R, M, C);
  return out;
}
