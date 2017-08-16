#include "meshgrid.cu"

class Flow_t
{
  public : 
    Flow_t() {}
    Flow_t(dim3 _gridsize, dim3 _blocksize, 
           data_t _t0, data_t _v0, data_t _D, data_t _rho0, data_t _P0, data_t _T0, 
           data_t _eta, data_t _lambda, data_t _R, data_t _M, data_t _C); 
    meshgrid_t operator()(const meshgrid_t& m);
    void flow_to_device();

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

    //----- dr -----
    __host__ __device__
    data_t dr_vr(int i, int j) const;
    __host__ __device__
    data_t dr_vz(int i, int j) const;
    __host__ __device__
    data_t dr_rho(int i, int j) const;
    __host__ __device__
    data_t dr_P(int i, int j) const;
    __host__ __device__
    data_t dr_T(int i, int j) const;
    //----- dz -----
    __host__ __device__
    data_t dz_vr(int i, int j) const;
    __host__ __device__
    data_t dz_vz(int i, int j) const;
    __host__ __device__
    data_t dz_rho(int i, int j) const;
    __host__ __device__
    data_t dz_P(int i, int j) const;
    __host__ __device__
    data_t dz_T(int i, int j) const;
       
    //----- d2r -----
    __host__ __device__
    data_t d2r_vr(int i, int j) const;
    __host__ __device__
    data_t d2r_vz(int i, int j) const;
    __host__ __device__
    data_t d2r_T(int i, int j) const;
    __host__ __device__
    data_t d2r_P(int i, int j) const;
    __host__ __device__
    data_t d2z_vr(int i, int j) const;
    __host__ __device__
    data_t d2z_vz(int i, int j) const;
    __host__ __device__
    data_t d2z_T(int i, int j) const;
    __host__ __device__
    data_t d2z_P(int i, int j) const;

    __host__ __device__
    data_t vgrad_vr(int i, int j) const;
    __host__ __device__
    data_t vgrad_vz(int i, int j) const;
    __host__ __device__
    data_t vgrad_P(int i, int j) const;
    __host__ __device__
    data_t vgrad_T(int i, int j) const;
    __host__ __device__
    data_t lapsca_P(int i, int j) const;
    __host__ __device__
    data_t lapsca_T(int i, int j) const;
    __host__ __device__
    data_t lapvec_vr(int i, int j) const;
    __host__ __device__
    data_t lapvec_vz(int i, int j) const;

  private:
    Flow_t *device_copy;
    data_t t0, v0, D, rho0, P0, T0, 
           eta, lambda,
           R, M, C, 
           alpha, beta, gamma, 
           delta, epsilon, zeta;
    meshgrid_t in, out;
    dim3 gridsize, blocksize;
};

//----- update -----
__global__ 
void one_step(Flow_t* F_pt)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;

  Flow_t& F = *F_pt;
  if (i>F.get_in().size_i || j>F.get_in().size_j) return;

  cell_t& out_c = F.get_out().dat(i, j);
  const cell_t& in_c = F.get_in().cdat(i, j);
  const data_t alpha = F.get_alpha(),
               beta = F.get_beta(),
               gamma = F.get_gamma(),
               delta = F.get_delta(),
               epsilon = F.get_epsilon(),
               zeta = F.get_zeta(),
               eta = F.get_eta(),
               lambda = F.get_lambda(),
               R = F.get_R(),
               M = F.get_M(),
               C = F.get_C();

  data_t rmc = R/(M*C),
         inv_rmc = 1./(1-rmc);

  out_c.vr = alpha * F.vgrad_vr(i, j)
           + beta * eta / in_c.rho * F.lapvec_vr(i, j)
           - gamma / in_c.rho * F.dr_P(i, j);

  out_c.vz = alpha * F.vgrad_vz(i, j)
           + beta * eta / in_c.rho * F.lapvec_vz(i, j)
           - gamma / in_c.rho * F.dz_P(i, j);

  out_c.rho = alpha * ( in_c.rho * in_c.vr * ( (data_t) F.get_in().size_j/j )
            + in_c.vr * F.dr_rho(i, j) + in_c.rho * F.dr_vr(i, j)
            + in_c.vz * F.dz_rho(i, j) + in_c.rho * F.dz_vz(i, j) );
  
  out_c.T = inv_rmc * rmc * out_c.rho 
          + delta * inv_rmc * eta * ( F.lapvec_vr(i, j) * in_c.vr + F.lapvec_vz(i, j) * in_c.vz )
          + epsilon * inv_rmc * lambda * F.lapsca_T(i, j); 

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
  cudaMalloc(&device_copy, sizeof(Flow_t));
}

meshgrid_t Flow_t::operator()(const meshgrid_t& m)
{
  out.from_meshgrid_prop(m);
  in = m;
  one_step<<<gridsize, blocksize>>>(device_copy);
  return out;
}

void Flow_t::flow_to_device() 
{
  in.is_on_device = true;
  out.is_on_device = true;
  cudaMemcpy(device_copy, this, sizeof(Flow_t), cudaMemcpyHostToDevice);
  in.is_on_device = false;
  out.is_on_device = false;
}

//----- dr -----
__host__ __device__
data_t Flow_t::dr_vr(int i, int j) const
{
  data_t tv, bv;
  if (j==0) return 0;
  else
  {
    if (in.cdat(i, j-1).is_wall) bv = 0;
    else bv = in.cdat(i, j-1).vr;
  }
  if (j==in.size_j-1) tv = 0;
  else
  {
    if (in.cdat(i, j+1).is_wall) tv = 0;
    else tv = in.cdat(i, j+1).vr;
  }
  return in.size_j*(tv-bv)/2;
}

__host__ __device__
data_t Flow_t::dr_vz(int i, int j) const
{
  data_t tv, bv;
  if (j==0) return 0;
  else
  {
    if (in.cdat(i, j-1).is_wall) bv = 0;
    else bv = in.cdat(i, j-1).vz;
  }
  if (j==in.size_j-1) tv = 0;
  else
  {
    if (in.cdat(i, j+1).is_wall) tv = 0;
    else tv = in.cdat(i, j+1).vz;
  }
  return in.size_j*(tv-bv)/2;
}

__host__ __device__
data_t Flow_t::dr_rho(int i, int j) const
{
  data_t tv, bv;
  if (j==0) return 0;
  else
  {
    if (in.cdat(i, j-1).is_wall) return 0;
    else bv = in.cdat(i, j-1).rho;
  }
  if (j==in.size_j-1) tv = 0;
  else
  {
    if (in.cdat(i, j+1).is_wall) return 0;
    else tv = in.cdat(i, j+1).rho;
  }
  return in.size_j*(tv-bv)/2;
}

__host__ __device__
data_t Flow_t::dr_P(int i, int j) const
{
  data_t tv, bv;
  if (j==0) return 0;
  else
  {
    if (in.cdat(i, j-1).is_wall) return 0;
    else bv = in.cdat(i, j-1).P;
  }
  if (j==in.size_j-1) tv = 0;
  else
  {
    if (in.cdat(i, j+1).is_wall) return 0;
    else tv = in.cdat(i, j+1).P;
  }
  return in.size_j*(tv-bv)/2;
}

__host__ __device__
data_t Flow_t::dr_T(int i, int j) const
{
  data_t tv, bv;
  if (j==0) return 0;
  else
  {
    if (in.cdat(i, j-1).is_wall) return 0;
    else bv = in.cdat(i, j-1).T;
  }
  if (j==in.size_j-1) tv = 0;
  else
  {
    if (in.cdat(i, j+1).is_wall) return 0;
    else tv = in.cdat(i, j+1).T;
  }
  return in.size_j*(tv-bv)/2;
}

//----- dz -----

__host__ __device__
data_t Flow_t::dz_vr(int i, int j) const
{
  data_t lv, rv;
  if (i==0) lv = 0;
  else
  {
    if (in.cdat(i-1, j).is_wall) lv = 0;
    else lv = in.cdat(i-1, j).vr;
  }
  if (i==in.size_i-1) rv = 0;
  else
  {
    if (in.cdat(i+1, j).is_wall) rv = 0;
    else rv = in.cdat(i+1, j).vr;
  }
  return in.size_i*(rv-lv)/2;
}
    
__host__ __device__
data_t Flow_t::dz_vz(int i, int j) const
{
  data_t lv, rv;
  if (i==0) lv = 0;
  else
  {
    if (in.cdat(i-1, j).is_wall) lv = 0;
    else lv = in.cdat(i-1, j).vz;
  }
  if (i==in.size_i-1) rv = 0;
  else
  {
    if (in.cdat(i+1, j).is_wall) rv = 0;
    else rv = in.cdat(i+1, j).vz;
  }
  return in.size_i*(rv-lv)/2;
}
    
__host__ __device__
data_t Flow_t::dz_rho(int i, int j) const
{
  data_t lv, rv;
  if (i==0) lv = 0;
  else
  {
    if (in.cdat(i-1, j).is_wall) return 0; 
    else lv = in.cdat(i-1, j).rho;
  }
  if (i==in.size_i-1) rv = 0;
  else
  {
    if (in.cdat(i+1, j).is_wall) return 0; 
    else rv = in.cdat(i+1, j).vz;
  }
  return in.size_i*(rv-lv)/2;
}
    
__host__ __device__
data_t Flow_t::dz_P(int i, int j) const
{
  data_t lv, rv;
  if (i==0) lv = 0;
  else
  {
    if (in.cdat(i-1, j).is_wall) return 0;
    else lv = in.cdat(i-1, j).vz;
  }
  if (i==in.size_i-1) rv = 0;
  else
  {
    if (in.cdat(i+1, j).is_wall) return 0;
    else rv = in.cdat(i+1, j).vz;
  }
  return in.size_i*(rv-lv)/2;
}
    
__host__ __device__
data_t Flow_t::dz_T(int i, int j) const
{
  data_t lv, rv;
  if (i==0) lv = 0;
  else
  {
    if (in.cdat(i-1, j).is_wall) return 0;
    else lv = in.cdat(i-1, j).vz;
  }
  if (i==in.size_i-1) rv = 0;
  else
  {
    if (in.cdat(i+1, j).is_wall) return 0;
    else rv = in.cdat(i+1, j).vz;
  }
  return in.size_i*(rv-lv)/2;
}
   
//----- d2r -----

__host__ __device__
data_t Flow_t::d2r_vr(int i, int j) const
{
  data_t tv, bv;
  if (j==0) bv=in.cdat(i,j).vr;
  else
  {
    if (in.cdat(i, j-1).is_wall) bv = 0;
    else bv = in.cdat(i, j-1).vr;
  }
  if (j==in.size_j-1) tv = in.cdat(i,j).vr;
  else
  {
    if (in.cdat(i, j+1).is_wall) tv = 0;
    else tv = in.cdat(i, j+1).vr;
  }
  return in.size_j*(tv+bv-2*in.cdat(i,j).vr);
}

__host__ __device__
data_t Flow_t::d2r_vz(int i, int j) const
{
  data_t tv, bv;
  if (j==0) bv=in.cdat(i,j).vz;
  else
  {
    if (in.cdat(i, j-1).is_wall) bv = 0;
    else bv = in.cdat(i, j-1).vz;
  }
  if (j==in.size_j-1) tv = in.cdat(i,j).vz;
  else
  {
    if (in.cdat(i, j+1).is_wall) tv = 0;
    else tv = in.cdat(i, j+1).vz;
  }
  return in.size_j*(tv+bv-2*in.cdat(i,j).vz);
}

__host__ __device__
data_t Flow_t::d2r_T(int i, int j) const
{
  data_t tv, bv;
  if (j==0) bv=in.cdat(i,j).T;
  else
  {
    if (in.cdat(i, j-1).is_wall) bv = in.cdat(i,j).T;
    else bv = in.cdat(i, j-1).vz;
  }
  if (j==in.size_j-1) tv = in.cdat(i,j).T;
  else
  {
    if (in.cdat(i, j+1).is_wall) tv = in.cdat(i,j).T;
    else tv = in.cdat(i, j+1).vz;
  }
  return in.size_j*(tv+bv-2*in.cdat(i,j).T);
}

__host__ __device__
data_t Flow_t::d2r_P(int i, int j) const
{
  data_t tv, bv;
  if (j==0) bv=in.cdat(i,j).P;
  else
  {
    if (in.cdat(i, j-1).is_wall) bv = in.cdat(i,j).P;
    else bv = in.cdat(i, j-1).P;
  }
  if (j==in.size_j-1) tv = in.cdat(i,j).P;
  else
  {
    if (in.cdat(i, j+1).is_wall) tv = in.cdat(i,j).P;
    else tv = in.cdat(i, j+1).vz;
  }
  return in.size_j*(tv+bv-2*in.cdat(i,j).P);
}

__host__ __device__
data_t Flow_t::d2z_vr(int i, int j) const
{
  data_t rv, lv;
  if (i==0) lv=in.cdat(i,j).vr;
  else
  {
    if (in.cdat(i-1, j).is_wall) lv = 0;
    else lv = in.cdat(i-1, j).vr;
  }
  if (i==in.size_i-1) rv = in.cdat(i,j).vr;
  else
  {
    if (in.cdat(i+1, j).is_wall) rv = 0;
    else rv = in.cdat(i+1, j).vr;
  }
  return in.size_j*(rv+lv-2*in.cdat(i,j).vr);
}

__host__ __device__
data_t Flow_t::d2z_vz(int i, int j) const
{
  data_t rv, lv;
  if (i==0) lv=in.cdat(i,j).vz;
  else
  {
    if (in.cdat(i-1, j).is_wall) lv = 0;
    else lv = in.cdat(i-1, j).vz;
  }
  if (i==in.size_i-1) rv = in.cdat(i,j).vz;
  else
  {
    if (in.cdat(i+1, j).is_wall) rv = 0;
    else rv = in.cdat(i+1, j).vz;
  }
  return in.size_j*(rv+lv-2*in.cdat(i,j).vz);
}

__host__ __device__
data_t Flow_t::d2z_T(int i, int j) const
{
  data_t rv, lv;
  if (i==0) lv=in.cdat(i,j).T;
  else
  {
    if (in.cdat(i-1, j).is_wall) lv = 0;
    else lv = in.cdat(i-1, j).T;
  }
  if (i==in.size_i-1) rv = in.cdat(i,j).T;
  else
  {
    if (in.cdat(i+1, j).is_wall) rv = 0;
    else rv = in.cdat(i+1, j).T;
  }
  return in.size_j*(rv+lv-2*in.cdat(i,j).T);
}

__host__ __device__
data_t Flow_t::d2z_P(int i, int j) const
{
  data_t rv, lv;
  if (i==0) lv=in.cdat(i,j).vz;
  else
  {
    if (in.cdat(i-1, j).is_wall) lv = 0;
    else lv = in.cdat(i-1, j).P;
  }
  if (i==in.size_i-1) rv = in.cdat(i,j).P;
  else
  {
    if (in.cdat(i+1, j).is_wall) rv = 0;
    else rv = in.cdat(i+1, j).P;
  }
  return in.size_j*(rv+lv-2*in.cdat(i,j).P);
}


__host__ __device__
data_t Flow_t::vgrad_vr(int i, int j) const
{
  return in.cdat(i,j).vr*dr_vr(i,j)+in.cdat(i,j).vz*dz_vr(i,j);
}

__host__ __device__
data_t Flow_t::vgrad_vz(int i, int j) const
{
  return in.cdat(i,j).vr*dr_vz(i,j)+in.cdat(i,j).vz*dz_vz(i,j);
}

__host__ __device__
data_t Flow_t::vgrad_P(int i, int j) const
{
  return in.cdat(i,j).vr*dr_P(i,j)+in.cdat(i,j).vz*dz_P(i,j);
}

__host__ __device__
data_t Flow_t::vgrad_T(int i, int j) const
{
  return in.cdat(i,j).vr*dr_T(i,j)+in.cdat(i,j).vz*dz_T(i,j);
}

__host__ __device__
data_t Flow_t::lapsca_P(int i, int j) const
{
  data_t val = d2r_P(i, j)+d2z_P(i, j);
  if (j>0) val += (data_t) in.size_j/j*dr_P(i, j);
  return val;
}

__host__ __device__
data_t Flow_t::lapsca_T(int i, int j) const
{
  data_t val = d2r_T(i, j)+d2z_T(i, j);
  if (j>0) val += (data_t) in.size_j/j*dr_T(i, j);
  return val;
}

__host__ __device__
data_t Flow_t::lapvec_vr(int i, int j) const
{
  data_t val = d2r_vr(i, j)+d2z_vr(i, j);
  if (j>0) val += (data_t) in.size_j/j*dr_vr(i, j) - 
                  (data_t) (in.size_j*in.size_j)/(j*j)*in.cdat(i,j).vr; //vr(r=0, z) = 0 !!!!!!
  return val;
}

__host__ __device__
data_t Flow_t::lapvec_vz(int i, int j) const
{
  data_t val = d2r_vz(i, j)+d2z_vz(i, j);
  if (j>0) val += (data_t) in.size_j/j*dr_vz(i, j);
  return val;
}
