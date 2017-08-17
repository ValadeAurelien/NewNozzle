#include <stdio.h>
#include <iostream>
#define CUDA_SAFE_CALL(call) \
  do { \
    cudaError_t err = call; \
      if (cudaSuccess != err) { \
        fprintf (stderr, "Cuda error in file '%s' in line %i : %s.", \
              __FILE__, __LINE__, cudaGetErrorString(err) ); \
       exit(EXIT_FAILURE); }} while (0)

//typedef float data_t;
typedef double data_t;
using namespace std; 

struct cell_t
{
  data_t vr = 0, 
         vz = 0, 
         P = 0, 
         rho = 0, 
         T = 0;
  bool is_wall = false;

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

__host__
ostream& operator<<(ostream& os, const cell_t& c)
{
  os << "(" << c.vr << ", " << c.vz << ", " << c.rho << ", " << c.T << ", " << c.P << ")";
  return os;
}

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

struct dev_meshgrid_t
{
  int size_i, size_j;
  cell_t* d_data = NULL;

//  ~dev_meshgrid_t() {}

  __device__ 
  const cell_t& cdat(int i, int j) const
  {
    if (i<size_i && j<size_j)
      return d_data[j*size_i+i];
    else 
    {
      printf("out of range : %d %d\n", i, j);
      return d_data[0];
    }
  }
  __device__ 
  cell_t& dat(int i, int j) 
  {
    if (i<size_i && j<size_j)
      return d_data[j*size_i+i];
    else 
    {
      printf("out of range : %d %d\n", i, j);
      return d_data[0];
    }
  }
};

struct meshgrid_t
{
  int size_i = 0, 
      size_j = 0;
  dim3 gridsize, blocksize;
  cell_t *h_data = NULL, 
         *d_data = NULL;
  dev_meshgrid_t *d_m = NULL;
  bool free_mem = true;

  __host__  
  meshgrid_t() {}
  __host__  
  meshgrid_t(const meshgrid_t& m);
  __host__  
  meshgrid_t(int _size_i, int _size_j, dim3 _gridsize, dim3 _blocksize);
  __host__ 
  ~meshgrid_t();
  __host__  
  void cpy_prop_new_data(const meshgrid_t& m); 
  __host__
  void reserve_host_data();

  __host__
  meshgrid_t operator+(const meshgrid_t& m) const;
  __host__
  meshgrid_t operator-(const meshgrid_t& m) const;
  __host__
  meshgrid_t operator*(const data_t& f) const;
  __host__
  meshgrid_t operator/(const data_t& f) const;
//  __host__
//  meshgrid_t& operator=(meshgrid_t&& m) noexcept;
  __host__
  meshgrid_t& operator=(const meshgrid_t& m);

  __host__ 
  data_t compare(const meshgrid_t& m) const;
  __host__ 
  bool equivalent(const meshgrid_t& m) const;

  __host__ 
  void data_to_host() ;
  __host__ 
  const cell_t& cat(int i, int j) const;
  __host__ 
  cell_t& at(int i, int j) ;
};

__host__
ostream& operator<<(ostream& os, const meshgrid_t& m)
{
  for (int j=0; j<m.size_j; j++)
  {
    for (int i=0; i<m.size_i; i++)
      cout << m.cat(i, j) << " ";
    cout << endl;
  }
  return os;
}

__host__
meshgrid_t::meshgrid_t(const meshgrid_t& m)
{
  *this = m;
}

__host__  
meshgrid_t::meshgrid_t(int _size_i, int _size_j, dim3 _gridsize, dim3 _blocksize) :
  size_i(_size_i), size_j(_size_j), gridsize(_gridsize), blocksize(_blocksize)
{
  CUDA_SAFE_CALL(cudaMalloc(&d_data, sizeof(cell_t)*size_i*size_j));
  if (!d_data) printf("could not allocate device memory (init, data)\n");

  dev_meshgrid_t tmp;
  tmp.size_i = size_i;
  tmp.size_j = size_j;
  tmp.d_data = d_data;
  CUDA_SAFE_CALL(cudaMalloc(&d_m, sizeof(dev_meshgrid_t)));
  if (!d_m) printf("could not allocate device memory (init, dm)\n");
  CUDA_SAFE_CALL(cudaMemcpy(d_m, &tmp, sizeof(dev_meshgrid_t), cudaMemcpyHostToDevice));
}

__host__ 
meshgrid_t::~meshgrid_t()
{
  if (free_mem)
  {
//    printf("d_data free (this)%p (device)%p (host)%p\n", this, d_data, h_data);
    if (d_data) CUDA_SAFE_CALL(cudaFree(d_data));
    if (d_m) CUDA_SAFE_CALL(cudaFree(d_m));
    if (h_data) free(h_data);
  }
}

__host__  
void meshgrid_t::cpy_prop_new_data(const meshgrid_t& m) 
{
  gridsize = m.gridsize;
  blocksize = m.blocksize;
  if(size_i != m.size_i || size_j != m.size_j)
  {
    size_i = m.size_i;
    size_j = m.size_j;
    if (d_data) CUDA_SAFE_CALL(cudaFree(d_data));
    CUDA_SAFE_CALL(cudaMalloc(&d_data, sizeof(cell_t)*size_i*size_j));
    if (!d_data) printf("could not allocate device memory (cpy, data)\n");
    if (h_data) h_data = (cell_t*) realloc(h_data, sizeof(size_i*size_j));

    dev_meshgrid_t tmp;
    tmp.size_i = size_i;
    tmp.size_j = size_j;
    tmp.d_data = d_data;
    if (!d_m) CUDA_SAFE_CALL(cudaMalloc(&d_m, sizeof(dev_meshgrid_t)));
    if (!d_m) printf("could not allocate device memory (cpy, dm)\n");
    CUDA_SAFE_CALL(cudaMemcpy(d_m, &tmp, sizeof(dev_meshgrid_t), cudaMemcpyHostToDevice));
  }
}

__host__
void meshgrid_t::reserve_host_data()
{
  if (!h_data) h_data = (cell_t*) malloc(sizeof(cell_t)*size_i*size_j); 
}

__host__
meshgrid_t meshgrid_t::operator+(const meshgrid_t& m) const
{ 
  if (!equivalent(m)) printf("meshgrids not equivalent\n"); 
  meshgrid_t res(size_i, size_j, gridsize, blocksize);
  cuda_operator_plus<<<gridsize, blocksize>>>(res.d_data, d_data, m.d_data, size_i, size_j);
  return res;
}

__host__
meshgrid_t meshgrid_t::operator-(const meshgrid_t& m) const 
{ 
  if (!equivalent(m)) printf("meshgrids not equivalent\n"); 
  meshgrid_t res(size_i, size_j, gridsize, blocksize);
  cuda_operator_minus<<<gridsize, blocksize>>>(res.d_data, d_data, m.d_data, size_i, size_j);
  return res;
}

__host__
meshgrid_t meshgrid_t::operator*(const data_t& f) const 
{ 
  meshgrid_t res(size_i, size_j, gridsize, blocksize);
  cuda_operator_times<<<gridsize, blocksize>>>(res.d_data, d_data, f, size_i, size_j);
  return res;
}

__host__
meshgrid_t meshgrid_t::operator/(const data_t& f) const 
{ 
  meshgrid_t res(size_i, size_j, gridsize, blocksize);
  cuda_operator_divided<<<gridsize, blocksize>>>(res.d_data, d_data, f, size_i, size_j);
  return res;
}

//__host__
//meshgrid_t& meshgrid_t::operator=(meshgrid_t&& m) noexcept
//{
//  printf("moving %p to %p\n", &m, this);
//  if (this == &m) return *this;
//  std::copy(&m, &m+sizeof(meshgrid_t), this);
//  m.d_data = NULL;
//  m.d_m = NULL;
//  m.h_data = NULL;
//  return *this;
//}

__host__
meshgrid_t& meshgrid_t::operator=(const meshgrid_t& m) 
{
//  printf("copying %p to %p\n", &m, this);
  if (this == &m) return *this;
  cpy_prop_new_data(m);
  CUDA_SAFE_CALL(cudaMemcpy(d_data, m.d_data, sizeof(cell_t)*size_i*size_j, cudaMemcpyDeviceToDevice));
  return *this;
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
  CUDA_SAFE_CALL(cudaMemcpy((void*) h_data, (const void*) d_data, (size_t) sizeof(cell_t)*size_i*size_j, cudaMemcpyDeviceToHost)); 
}

__host__ 
const cell_t& meshgrid_t::cat(int i, int j) const
{
  if (i<size_i && j<size_j)
    return h_data[j*size_i+i];
  else 
  {
    printf("out of range : %d %d\n", i, j);
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
    printf("out of range : %d %d\n", i, j);
    return h_data[0];
  }
}


