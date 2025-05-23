#ifndef _CUDA_HELPERS_INCLUDED_
#define _CUDA_HELPERS_INCLUDED_

#include <cstring>
#if defined (CAMP_HAVE_CUDA)
#include <cuda_runtime.h>
#endif

#if defined (ASCENT_CUDA_ENABLED)
#include <cuda_runtime.h>
#endif

/*
code inspired from ascent/src/tests/ascent/t_ascent_gpu_data_source.cpp
*/

static void cuda_check_status(cudaError_t status) {
    if(status != cudaSuccess) {
        std::cerr << "error: CUDA API call : "
                  << cudaGetErrorString(status) << std::endl;
        exit(1);
    }
}

//-----------------------------------------------------------------------------
void *
device_alloc(int size)
{
#if defined (ASCENT_CUDA_ENABLED)
  void *buff;
  auto status = cudaMalloc(&buff, size);
  cuda_check_status(status);
  return buff;
#else
  return nullptr;
#endif
}

//-----------------------------------------------------------------------------
void
device_free(void *ptr)
{
#if defined (CAMP_HAVE_CUDA)
  cudaFree(ptr);
  free(ptr);
#endif
}

//-----------------------------------------------------------------------------
void
copy_from_device_to_host(void *dest, void *src, int size)
{
#if defined (ASCENT_CUDA_ENABLED)
  cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
  memcpy(dest,src,size);
#endif
}


//-----------------------------------------------------------------------------
void
copy_from_host_to_device(void *dest, void *src, int size)
{
#if defined (ASCENT_CUDA_ENABLED)
  auto status = cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
  //memcpy(dest,src,size);
  cuda_check_status(status);
#endif
}

//-----------------------------------------------------------------------------
void
device_move(conduit::Node &data, int data_nbytes)
{
  // alloc proper size
  //conduit::index_t data_nbytes = data.total_bytes_allocated();
  void *device_ptr = device_alloc(data_nbytes);
  copy_from_host_to_device(device_ptr, data.data_ptr(), data_nbytes);
  conduit::DataType dtype = data.dtype();
  data.set_external(dtype,device_ptr);
}

#endif
