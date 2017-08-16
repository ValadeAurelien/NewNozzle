#include <stdio.h>
typedef float data_t;

struct cell_t
{
  data_t vr, vz, P, rho, T;
  bool is_wall;

  __host__ __device__
  cell_t(data_t _vr, data_t _vz, data_t _P, data_t _rho, data_t _T, bool _is_wall) : vr(_vr), vz(_vz), P(_P), rho(_rho), T(_T), is_wall(_is_wall) {}

  __host__ __device__
  cell_t operator+(const cell_t& c) { return cell_t(vr+c.vr, vz+c.vz, P+c.P, rho+c.rho, T+c.T, is_wall); }
  __host__ __device__
  cell_t operator-(const cell_t& c) { return cell_t(vr-c.vr, vz-c.vz, P-c.P, rho-c.rho, T-c.T, is_wall); }
  __host__ __device__
  cell_t operator*(const data_t& f) { return cell_t(vr*f, vz*f, P*f, rho*f, T*f, is_wall); }
  __host__ __device__
  cell_t operator/(const data_t& f) { return cell_t(vr/f, vz/f, P/f, rho/f, T/f, is_wall); }
};

__global__
void cuda_operator_plus(cell_t* C, cell_t* A, cell_t* B, int size_i, int size_j)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  if (i>=size_i || j>=size_j) return;
  C[j*size_i+i] = A[j*size_i+i] + B[j*size_i+i];
}
__global__
void cuda_operator_minus(cell_t* C, cell_t* A, cell_t* B, int size_i, int size_j)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  if (i>=size_i || j>=size_j) return;
  C[j*size_i+i] = A[j*size_i+i] - B[j*size_i+i];
}
__global__
void cuda_operator_times(cell_t* B, cell_t* A, data_t f, int size_i, int size_j)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  if (i>=size_i || j>=size_j) return;
  B[j*size_i+i] = A[j*size_i+i] * f;
}
__global__
void cuda_operator_divided(cell_t* B, cell_t* A, data_t f, int size_i, int size_j)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int j = blockIdx.y*blockDim.y+threadIdx.y;
  if (i>=size_i || j>=size_j) return;
  B[j*size_i+i] = A[j*size_i+i] / f;
}

struct meshgrid_t
{
  int size_i, size_j;
  dim3 gridsize, blocksize;
  cell_t* h_data, *d_data;
  bool is_on_device=false;

  __host__  
  meshgrid_t() {}
  __host__  
  meshgrid_t(int _size_i, int _size_j, dim3 _gridsize, dim3 _blocksize);
  __host__ 
  ~meshgrid_t();
  __host__  
  void from_meshgrid_prop(const meshgrid_t& m); 
  __host__  
  meshgrid_t* meshgrid_to_device(); 

  __host__
  meshgrid_t operator+(const meshgrid_t& m) const;
  __host__
  meshgrid_t operator-(const meshgrid_t& m) const;
  __host__
  meshgrid_t operator*(const data_t& f) const;
  __host__
  meshgrid_t operator/(const data_t& f) const;

  __host__ 
  data_t compare(const meshgrid_t& m) const;
  __host__ 
  bool equivalent(const meshgrid_t& m) const;

  __host__ 
  void data_to_host();
  __host__ 
  const cell_t& cat(int i, int j) const;
  __host__ 
  cell_t& at(int i, int j) ;
  __host__ __device__ 
  const cell_t& cdat(int i, int j) const;
  __host__ __device__ 
  cell_t& dat(int i, int j) ;
};

__host__  
meshgrid_t::meshgrid_t(int _size_i, int _size_j, dim3 _gridsize, dim3 _blocksize) :
  size_i(_size_i), size_j(_size_j), gridsize(_gridsize), blocksize(_blocksize)
{
  cudaMalloc(&d_data, sizeof(cell_t)*size_i*size_j);
  if (!d_data) printf("could not allocate device memory");
}

__host__ 
meshgrid_t::~meshgrid_t()
{
  if (!is_on_device) 
  {
    cudaFree(&d_data);
//    if (h_data!=NULL) free(h_data);
  }
}

__host__  
void meshgrid_t::from_meshgrid_prop(const meshgrid_t& m) 
{
  size_i = m.size_i;
  size_j = m.size_j;
  gridsize = m.gridsize;
  blocksize = m.blocksize;
  if (!d_data) cudaMalloc(&d_data, sizeof(cell_t)*size_i*size_j);
}

__host__
meshgrid_t* meshgrid_t::meshgrid_to_device()
{
  meshgrid_t* cpy;
  cudaMalloc(&cpy, sizeof(meshgrid_t));
  is_on_device = true;
  cudaMemcpy(cpy, this, sizeof(meshgrid_t), cudaMemcpyHostToDevice);
  if (!cpy) printf("could not allocate device memory");
  is_on_device = false;
  return cpy;
}

__host__
meshgrid_t meshgrid_t::operator+(const meshgrid_t& m) const
{ 
  if (!equivalent(m)) printf("meshgrids not equivalent"); 
  meshgrid_t res(size_i, size_j, gridsize, blocksize);
  cuda_operator_plus<<<blocksize, gridsize>>>(res.d_data, d_data, m.d_data, size_i, size_j);
  return res;
}

__host__
meshgrid_t meshgrid_t::operator-(const meshgrid_t& m) const 
{ 
  if (!equivalent(m)) printf("meshgrids not equivalent"); 
  meshgrid_t res(size_i, size_j, gridsize, blocksize);
  cuda_operator_minus<<<blocksize, gridsize>>>(res.d_data, d_data, m.d_data, size_i, size_j);
  return res;
}

__host__
meshgrid_t meshgrid_t::operator*(const data_t& f) const 
{ 
  meshgrid_t res(size_i, size_j, gridsize, blocksize);
  cuda_operator_times<<<blocksize, gridsize>>>(res.d_data, d_data, f, size_i, size_j);
  return res;
}

__host__
meshgrid_t meshgrid_t::operator/(const data_t& f) const 
{ 
  meshgrid_t res(size_i, size_j, gridsize, blocksize);
  cuda_operator_divided<<<blocksize, gridsize>>>(res.d_data, d_data, f, size_i, size_j);
  return res;
}

__host__ 
data_t meshgrid_t::compare(const meshgrid_t& m) const
{
  return 0;
}

__host__ 
bool meshgrid_t::equivalent(const meshgrid_t& m) const
{
  return (size_i==m.size_i && size_j==m.size_j && 
          gridsize.x==m.gridsize.x && blocksize.x==m.blocksize.x &&
          gridsize.y==m.gridsize.y && blocksize.y==m.blocksize.y);
}

__host__ 
void meshgrid_t::data_to_host()
{
  h_data = (cell_t*) malloc(sizeof(cell_t)*size_i*size_j);
  cudaMemcpy(h_data, d_data, sizeof(cell_t)*size_i*size_j, cudaMemcpyDeviceToHost); 
}

__host__ 
const cell_t& meshgrid_t::cat(int i, int j) const
{
  if (i<size_i && j<size_j)
    return h_data[j*size_i+i];
  else 
  {
    printf("out of range");
    return h_data[0];
  }
}

  __host__ 
cell_t& meshgrid_t::at(int i, int j) 
{
  if (i<size_i && j<size_j)
    return h_data[j*size_i+i];
  else 
  {
    printf("out of range");
    return h_data[0];
  }
}

__host__ __device__ 
const cell_t& meshgrid_t::cdat(int i, int j) const
{
  if (i<size_i && j<size_j)
    return d_data[j*size_i+i];
  else 
  {
    printf("out of range");
    return d_data[0];
  }
}

__host__ __device__ 
cell_t& meshgrid_t::dat(int i, int j) 
{
  if (i<size_i && j<size_j)
    return d_data[j*size_i+i];
  else 
  {
    printf("out of range");
    return d_data[0];
  }
}
