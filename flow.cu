#include "meshgrid.cu"

class Flow_t
{
  public : 
    Flow_t() {}
    Flow_t(dim3 _gridsize, dim3 _blocksize, data_t _R, data_t _M, data_t _C, data_t _eta); 
 

    meshgrid_t operator()(const meshgrid_t& m);
 
  private :
    void update_vr();
    void update_vz();
    void update_rho();
    void update_T();
    void update_P();

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

    __host__ __device__
    data_t dz_vr(int i, int j) const;
    __host__ __device__
    data_t dz_vr(int i, int j) const;
    __host__ __device__
    data_t dz_rho(int i, int j) const;
    __host__ __device__
    data_t dz_P(int i, int j) const;
    __host__ __device__
    data_t dz_T(int i, int j) const;

    __host__ __device__
    data_t d2r_vr(int i, int j) const;
    __host__ __device__
    data_t d2r_vz(int i, int j) const;
    __host__ __device__
    data_t d2r_T(int i, int j) const;
    __host__ __device__
    data_t d2z_vr(int i, int j) const;
    __host__ __device__
    data_t d2z_vz(int i, int j) const;
    __host__ __device__
    data_t d2z_T(int i, int j) const;

    __host__ __device__
    data_t vgrad_vr(int i, int j) const;
    __host__ __device__
    data_t vgrad_vz(int i, int j) const;
    __host__ __device__
    data_t vgrad_P(int i, int j) const;
    __host__ __device__
    data_t vgrad_T(int i, int j) const;
    __host__ __device__
    data_t lapsac_P(int i, int j) const;
    __host__ __device__
    data_t lapsca_T(int i, int j) const;
    __host__ __device__
    data_t lapvec_vr(int i, int j) const;
    __host__ __device__
    data_t lapvec_vz(int i, int j) const;

    data_t R, M, C, eta;
    meshgrid_t in, out;
    dim3 gridsize, blocksize;
};


Flow_t::Flow_t(dim3 _gridsize, dim3 _blocksize, data_t _R, data_t _M, data_t _C, data_t _eta)
  : gridsize(_gridsize), blocksize(_blocksize), R(_R), M(_M), C(_C), eta(_eta) 
{}

meshgrid_t Flow_t::operator()(const meshgrid_t& m)
{
  out.from_meshgrid_prop(m);
  in = m;
  update_vr();
  update_vz();
  update_rho();
  update_T();
  update_P();
  return out;
}


//----- dr -----
__host__ __device__
data_t Flow_t::dr_vr(int i, int j) const
{
  data_t tv, bv;
  if (j==0) return 0;
  else
  {
    if (in.dat(i, j-1).is_wall) bv = 0;
    else bv = in.dat(i, j-1).vr;
  }
  if (j==size_j-1) tv = 0;
  else
  {
    if (in.dat(i, j+1).is_wall) tv = 0;
    else tv = in.dat(i, j+1).vr;
  }
  return size_j*(tv-bv)/2;
}

__host__ __device__
data_t Flow_t::dr_vz(int i, int j) const
{
  data_t tv, bv;
  if (j==0) return 0;
  else
  {
    if (in.dat(i, j-1).is_wall) bv = 0;
    else bv = in.dat(i, j-1).vz;
  }
  if (j==size_j-1) tv = 0;
  else
  {
    if (in.dat(i, j+1).is_wall) tv = 0;
    else tv = in.dat(i, j+1).vz;
  }
  return size_j*(tv-bv)/2;
}

__host__ __device__
data_t Flow_t::dr_rho(int i, int j) const
{
  data_t tv, bv;
  if (j==0) return 0;
  else
  {
    if (in.dat(i, j-1).is_wall) return 0;
    else bv = in.dat(i, j-1).rho;
  }
  if (j==size_j-1) tv = 0;
  else
  {
    if (in.dat(i, j+1).is_wall) return 0;
    else tv = in.dat(i, j+1).rho;
  }
  return size_j*(tv-bv)/2;
}

__host__ __device__
data_t Flow_t::dr_P(int i, int j) const
{
  data_t tv, bv;
  if (j==0) return 0;
  else
  {
    if (in.dat(i, j-1).is_wall) return 0;
    else bv = in.dat(i, j-1).P;
  }
  if (j==size_j-1) tv = 0;
  else
  {
    if (in.dat(i, j+1).is_wall) return 0;
    else tv = in.dat(i, j+1).P;
  }
  return size_j*(tv-bv)/2;
}

__host__ __device__
data_t Flow_t::dr_T(int i, int j) const
{
  data_t tv, bv;
  if (j==0) return 0;
  else
  {
    if (in.dat(i, j-1).is_wall) return 0;
    else bv = in.dat(i, j-1).T;
  }
  if (j==size_j-1) tv = 0;
  else
  {
    if (in.dat(i, j+1).is_wall) return 0;
    else tv = in.dat(i, j+1).T;
  }
  return size_j*(tv-bv)/2;
}

//----- dz -----

__host__ __device__
data_t Flow_t::dz_vr(int i, int j) const
{
  data_t lv, rv;
  if (i==0) lv = 0;
  else
  {
    if (in.dat(i-1, j).is_wall) lv = 0;
    else lv = in.dat(i-1, j).vr;
  }
  if (i==size_i-1) rv = 0;
  else
  {
    if (in.dat(i+1, j).is_wall) rv = 0;
    else rv = in.dat(i+1, j).vr;
  }
  return size_i*(rv-lv)/2;
}
    
__host__ __device__
data_t Flow_t::dz_vz(int i, int j) const
{
  data_t lv, rv;
  if (i==0) lv = 0;
  else
  {
    if (in.dat(i-1, j).is_wall) lv = 0;
    else lv = in.dat(i-1, j).vz;
  }
  if (i==size_i-1) rv = 0;
  else
  {
    if (in.dat(i+1, j).is_wall) rv = 0;
    else rv = in.dat(i+1, j).vz;
  }
  return size_i*(rv-lv)/2;
}
    
__host__ __device__
data_t Flow_t::dz_rho(int i, int j) const
{
  data_t lv, rv;
  if (i==0) lv = 0;
  else
  {
    if (in.dat(i-1, j).is_wall) return 0; 
    else lv = in.dat(i-1, j).rho;
  }
  if (i==size_i-1) rv = 0;
  else
  {
    if (in.dat(i+1, j).is_wall) return 0; 
    else rv = in.dat(i+1, j).vz;
  }
  return size_i*(rv-lv)/2;
}
    
__host__ __device__
data_t Flow_t::dz_P(int i, int j) const
{
  data_t lv, rv;
  if (i==0) lv = 0;
  else
  {
    if (in.dat(i-1, j).is_wall) return 0;
    else lv = in.dat(i-1, j).vz;
  }
  if (i==size_i-1) rv = 0;
  else
  {
    if (in.dat(i+1, j).is_wall) return 0;
    else rv = in.dat(i+1, j).vz;
  }
  return size_i*(rv-lv)/2;
}
    
__host__ __device__
data_t Flow_t::dz_T(int i, int j) const
{
  data_t lv, rv;
  if (i==0) lv = 0;
  else
  {
    if (in.dat(i-1, j).is_wall) return 0;
    else lv = in.dat(i-1, j).vz;
  }
  if (i==size_i-1) rv = 0;
  else
  {
    if (in.dat(i+1, j).is_wall) return 0;
    else rv = in.dat(i+1, j).vz;
  }
  return size_i*(rv-lv)/2;
}
   
//----- d2r -----

__host__ __device__
data_t d2r_vr(int i, int j) const
{
  data_t tv, bv;
  if (j==0) bv=in.dat(i,j).vr;
  else
  {
    if (in.dat(i, j-1).is_wall) bv = 0;
    else bv = in.dat(i, j-1).vr;
  }
  if (j==size_j-1) tv = in.dat(i,j).vr;
  else
  {
    if (in.dat(i, j+1).is_wall) tv = 0;
    else tv = in.dat(i, j+1).vr;
  }
  return size_j*(tv+bv-2*in.dat(i,j));
}

__host__ __device__
data_t d2r_vz(int i, int j) const
{
  data_t tv, bv;
  if (j==0) bv=in.dat(i,j).vz;
  else
  {
    if (in.dat(i, j-1).is_wall) bv = 0;
    else bv = in.dat(i, j-1).vz;
  }
  if (j==size_j-1) tv = in.dat(i,j).vz;
  else
  {
    if (in.dat(i, j+1).is_wall) tv = 0;
    else tv = in.dat(i, j+1).vz;
  }
  return size_j*(tv+bv-2*in.dat(i,j));
}

__host__ __device__
data_t d2r_T(int i, int j) const
{
  data_t tv, bv;
  if (j==0) bv=in.dat(i,j).T;
  else
  {
    if (in.dat(i, j-1).is_wall) bv = in.dat(i,j).T;
    else bv = in.dat(i, j-1).vz;
  }
  if (j==size_j-1) tv = in.dat(i,j).T;
  else
  {
    if (in.dat(i, j+1).is_wall) tv = in.dat(i,j).T;
    else tv = in.dat(i, j+1).vz;
  }
  return size_j*(tv+bv-2*in.dat(i,j));
}

__host__ __device__
data_t d2z_vr(int i, int j) const
{
  data_t rv, lv;
  if (i==0) lv=in.dat(i,j).vr;
  else
  {
    if (in.dat(i-1, j).is_wall) lv = 0;
    else lv = in.dat(i-1, j).vr;
  }
  if (i==size_i-1) rv = in.dat(i,j).vr;
  else
  {
    if (in.dat(i+1, j).is_wall) rv = 0;
    else rv = in.dat(i+1, j).vr;
  }
  return size_j*(rv+lv-2*in.dat(i,j));
}

__host__ __device__
data_t d2z_vz(int i, int j) const
{
  data_t rv, lv;
  if (i==0) lv=in.dat(i,j).vz;
  else
  {
    if (in.dat(i-1, j).is_wall) lv = 0;
    else lv = in.dat(i-1, j).vz;
  }
  if (i==size_i-1) rv = in.dat(i,j).vz;
  else
  {
    if (in.dat(i+1, j).is_wall) rv = 0;
    else rv = in.dat(i+1, j).vz;
  }
  return size_j*(rv+lv-2*in.dat(i,j));
}

__host__ __device__
data_t d2z_T(int i, int j) const
{
  data_t rv, lv;
  if (i==0) lv=in.dat(i,j).vz;
  else
  {
    if (in.dat(i-1, j).is_wall) lv = 0;
    else lv = in.dat(i-1, j).T;
  }
  if (i==size_i-1) rv = in.dat(i,j).vz;
  else
  {
    if (in.dat(i+1, j).is_wall) rv = 0;
    else rv = in.dat(i+1, j).T;
  }
  return size_j*(rv+lv-2*in.dat(i,j));
}


__host__ __device__
data_t vgrad_vr(int i, int j) const
{
  return in.dat(i,j).vr*dr_vr(i,j)+in.dat(i,j).vz*dz_vr(i,j);
}

__host__ __device__
data_t vgrad_vz(int i, int j) const
{
  return in.dat(i,j).vr*dr_vz(i,j)+in.dat(i,j).vz*dz_vz(i,j);
}

__host__ __device__
data_t vgrad_P(int i, int j) const
{
  return in.dat(i,j).vr*dr_P(i,j)+in.dat(i,j).vz*dz_P(i,j);
}

__host__ __device__
data_t vgrad_T(int i, int j) const
{
  return in.dat(i,j).vr*dr_T(i,j)+in.dat(i,j).vz*dz_T(i,j);
}

__host__ __device__
data_t lapsca_P(int i, int j) const
{
  data_t val = d2r_P(i, j)+d2z_P(i, j);
  if (j>0) val += (data_t) size_j/j*dr_P(i, j);
  return val;
}

__host__ __device__
data_t lapsca_T(int i, int j) const
{
  data_t val = d2r_T(i, j)+d2z_T(i, j);
  if (j>0) val += (data_t) size_j/j*dr_T(i, j);
  return val;
}

__host__ __device__
vec2D_t lapvec_vr(int i, int j) const
{
  data_t val = d2r_vr(i, j)+d2z_vr(i, j);
  if (j>0) val += (data_t) size_j/j*dr_vr(i, j) - 
                  (data_t) (size_j*size_j)/(j*j)*in.dat(i,j).vr; //vr(r=0, z) = 0 !!!!!!
  return val;
}

__host__ __device__
vec2D_t lapvec_vr(int i, int j) const
{
  data_t val = d2r_vz(i, j)+d2z_vz(i, j);
  if (j>0) val += (data_t) size_j/j*dr_vz(i, j);
  return val;
}

//----- updates -----


__host__
void update_vr()
{

