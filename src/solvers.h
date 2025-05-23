#ifndef _SOLVERS_H_INCLUDED_
#define _SOLVERS_H_INCLUDED_

#include <vector>
#include <math.h>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#ifdef LOAD_H5Part
#include <hdf5.h>

static int ReadHDF5Dataset(const char *name, const hid_t mesh_id, void *data, int size)
{
  hid_t dset_id, filespace, attr1;
  herr_t  status;
  hsize_t dimsf[2]={0,0};

  dset_id = H5Dopen(mesh_id, name, H5P_DEFAULT);
  
  if(dset_id >= 0)
    {
    filespace = H5Dget_space(dset_id);
    H5Sget_simple_extent_dims(filespace, dimsf, NULL);
    H5Sclose(filespace);
  
    if (size == 4 && H5Dread(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0)
      {
      std::cerr << "Error reading dataset " << name << std::endl;
      return -1;
      }
    if (size == 8 && H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0)
      {
      std::cerr << "Error reading dataset " << name << std::endl;
      return -1;
      }
    }
    H5Dclose(dset_id);
    return dimsf[0]; // returns number of particles
}
#endif

namespace sph
{
template<typename T>
class ParticlesData
{
  public:
    int                 par_rank;
    int                 par_size;
    int                 iteration;       // Current iteration
    T                   time;
    size_t              n;               // Number of particles


#ifdef STRIDED_SCALARS
    struct tipsySph
    {
      T mass;
      T pos[3];
      T vel[3];
      T rho;
      T temp;
      T hsmooth;
      T metals;
      T phi;
    };
    std::vector<tipsySph> scalarsAOS;
    static constexpr int NbofScalarfields = sizeof(tipsySph)/sizeof(T);
#else
    std::vector<T> mass;            // "mass"
    std::vector<T> x, y, z;         // Positions
    std::vector<T> vx, vy, vz;      // Velocities
    std::vector<T> rho;             // Density
    std::vector<T> temp;            // Temperature

    static constexpr int NbofScalarfields = 9;
#endif

    static constexpr T cstMass = 0.12345;
    static constexpr T bbox_offset = 2.0; // to offset each MPI partition in 3D space
    
    ParticlesData()
    {
    MPI_Comm_rank(MPI_COMM_WORLD, &this->par_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->par_size);
    }
#ifdef LOAD_H5Part
    int UseH5PartData(const std::string  &H5PartFileName)
    {
    this->iteration = 0;
#ifdef STRIDED_SCALARS
    std::cerr << "Not implemented for strided data layout. Must reconfigure with cmake\n";
    exit(-1);
#endif
    hid_t file_id;
    file_id = H5Fopen(H5PartFileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if(file_id != H5I_INVALID_HID)
      {
      std::cout << __LINE__ << " :found valid HDF5 file " << H5PartFileName << std::endl;
      hid_t root_id = H5Gopen(file_id, "/", H5P_DEFAULT);
      if (root_id != H5I_INVALID_HID)
        {
        hid_t attr1 = H5Aopen_name(root_id, "numParticlesGlobal");
        if (H5Aread(attr1, H5T_NATIVE_INT, &this->n) < 0)
         {
         std::cerr << "missing attributes numParticlesGlobal" << std::endl;
         }
        else
          {
          H5Aclose(attr1);
          std::cout << __LINE__ << " :found " << this->n << " particles"<<std::endl;
          }
        hid_t step_id = H5Gopen(root_id, "Step#0", H5P_DEFAULT);
        if (step_id != H5I_INVALID_HID)
          {
          int N;
// N.B. x,y,z,temp are doubles, the others are float
// thus, in Driver.cxx you must use   ParticlesData<float> *sim = new(ParticlesData<float>);
          std::cout << __LINE__ << " :found valid HDF5 Step#0 " << std::endl;
          std::cout << "Allocating 9 std::vectors of scalar fields of size " << sizeof(T) << " bytes" << std::endl;

          this->x.resize(this->n);
          N = ReadHDF5Dataset("x", step_id, this->x.data(), sizeof(T));
          if(N != this->n) std::cerr << __LINE__ << "  Error reading dataset" << std::endl;
          
          this->y.resize(this->n);
          N = ReadHDF5Dataset("y", step_id, this->y.data(), sizeof(T));
          if(N != this->n) std::cerr << __LINE__ << "  Error reading dataset" << std::endl;
          
          this->z.resize(this->n);
          N = ReadHDF5Dataset("z", step_id, this->z.data(), sizeof(T));
          if(N != this->n) std::cerr << __LINE__ << "  Error reading dataset" << std::endl;
          
          this->vx.resize(this->n);
          N = ReadHDF5Dataset("vx", step_id, this->vx.data(), sizeof(T));
          if(N != this->n) std::cerr << __LINE__ << "  Error reading dataset" << std::endl;
          
          this->vy.resize(this->n);
          N = ReadHDF5Dataset("vy", step_id, this->vy.data(), sizeof(T));
          if(N != this->n) std::cerr << __LINE__ << "  Error reading dataset" << std::endl;
          
          this->vz.resize(this->n);
          N = ReadHDF5Dataset("vz", step_id, this->vz.data(), sizeof(T));
          if(N != this->n) std::cerr << __LINE__ << "  Error reading dataset" << std::endl;
          
          this->rho.resize(this->n); // using alpha since 'rho' does not exist currently
          N = ReadHDF5Dataset("rho", step_id, this->rho.data(), sizeof(T));
          if(N != this->n) std::cerr << __LINE__ << "  Error reading dataset" << std::endl;
          auto minIt = std::min_element(this->rho.begin(), this->rho.end());
          auto maxIt = std::max_element(this->rho.begin(), this->rho.end());
          std::cout << *minIt << " < rho < " << *maxIt << std::endl;
          
          this->mass.resize(this->n);
          N = ReadHDF5Dataset("m", step_id, this->mass.data(), sizeof(T));
          if(N != this->n) std::cerr << __LINE__ << "  Error reading dataset" << std::endl;
          
          this->temp.resize(this->n);
          N = ReadHDF5Dataset("temp", step_id, this->temp.data(), sizeof(T));
          if(N != this->n) std::cerr << __LINE__ << "  Error reading dataset" << std::endl;
          
          //std::cout << __LINE__ << " H5Gclose(step_id) " << std::endl;
          H5Gclose(step_id);
          }
        else
          {
          std::cerr << "cannot open /Step#0 " << std::endl;
          exit(-1);
          }
        //std::cout << __LINE__ << " H5Gclose(root_id) " << std::endl;
        H5Gclose(root_id);
        }
        //std::cout << __LINE__ << " H5Fclose(file_id) " << std::endl;
        H5Fclose(file_id);
      }
    else
      {
      std::cerr << "cannot open valid HDF5 file " << H5PartFileName << std::endl;
      exit(1);
      }
    //std::cout << __LINE__ << " finished loading H5Part data " << std::endl;
    return(0);
    };
#endif
    void UseTipsyData(const float *data, int N)
    {
    this->iteration = 0;
#ifdef STRIDED_SCALARS
    if(sizeof(T) == sizeof(float))
      {
      this->n = N;
      this->scalarsAOS.resize(this->n);
      memcpy(this->scalarsAOS.data(), data, N*sizeof(tipsySph));
      }
    else
      {
      std::cerr << "Not implemented for float64 data layout. Must reconfigure Driver.cxx\n";
      exit(-1);
      }
#else
    std::cerr << "Not implemented for non-strided data layout. Must reconfigure with cmake\n";
    exit(-1);
#endif
    };

    void AllocateGridMemory(int N)
    {
    this->n = N*N*N;
    this->iteration = 0;

#ifdef STRIDED_SCALARS
    this->scalarsAOS.resize(this->n);
#else
    this->x.resize(this->n);
    this->y.resize(this->n);
    this->z.resize(this->n);
    this->rho.resize(this->n);
    this->temp.resize(this->n);
    this->mass.resize(this->n);
    this->vx.resize(this->n);
    this->vy.resize(this->n);
    this->vz.resize(this->n);
#endif
    srand(1234 + this->par_rank);
    //std::cerr << "adding offset for coordinates " << sim.par_rank << std::endl;
    int id=0;
    for (auto iz=0; iz < N; iz++) {
      T zz = -1.0f + 2.0f*iz/(N - 1.0f);
      for (auto iy=0; iy < N; iy++){
        T yy = bbox_offset*this->par_rank - 1.0f + 2.0f*iy/(N - 1.0f);
        for (auto ix=0; ix < N; ix++)
          {
#ifdef STRIDED_SCALARS
          this->scalarsAOS[id].pos[0] = bbox_offset*this->par_rank -1.0f + 2.0f*ix/(N - 1.0);
          this->scalarsAOS[id].pos[1] = yy;
          this->scalarsAOS[id].pos[2] = zz;
#else
          this->x[id] = bbox_offset*this->par_rank -1.0f + 2.0f*ix/(N - 1.0);
          this->y[id] = yy;
          this->z[id] = zz;
#endif
          id++;
         }
      }
    }

    for (size_t i=0; i < this->n; i++)
      {
#ifdef STRIDED_SCALARS
      /* rho is equal to radius_square */
      T R = (this->scalarsAOS[i].pos[0] * this->scalarsAOS[i].pos[0] +
             this->scalarsAOS[i].pos[1] * this->scalarsAOS[i].pos[1] +
             this->scalarsAOS[i].pos[2] * this->scalarsAOS[i].pos[2]);
      this->scalarsAOS[i].rho = R;
      this->scalarsAOS[i].temp = sqrt(R);
      this->scalarsAOS[i].mass = cstMass;
      this->scalarsAOS[i].vel[0] = R;
      this->scalarsAOS[i].vel[1] = R;
      this->scalarsAOS[i].vel[2] = R;
#else
      T R = (this->x[i]*this->x[i] + /* rho is equal to radius_square */
             this->y[i]*this->y[i] +
             this->z[i]*this->z[i]);
      this->rho[i] = R;
      this->temp[i] = sqrt(R);
      this->mass[i] = cstMass;
      this->vx[i] = R;
      this->vy[i] = R;
      this->vz[i] = R;
#endif
      }
    //std::cout << "Vectors of size " << sim.x.size() << std::endl;
    };

    void FreeGridMemory()
    {
#ifdef STRIDED_SCALARS
    this->scalarsAOS.clear();
#else
    this->x.clear();
    this->y.clear();
    this->z.clear();
    this->rho.clear();
    this->temp.clear();
    this->mass.clear();
    this->vx.clear();
    this->vy.clear();
    this->vz.clear();
#endif
    };
    
    void simulate_one_timestep()
    {
    this->iteration++;
    this->time = this->iteration * 0.01; // fixed, arbitrary timestep value
    T timeoffset = 0.3*sin((this->iteration * M_PI) / 180.0); // fixed, arbitrary timestep value
    std::cout << "timeoffset " << timeoffset << std::endl;
    // all variables remain constant over time except "rho" and "temp"
    for (size_t i=0; i < this->n; i++)
      {
#ifdef STRIDED_SCALARS
      T R = ((this->scalarsAOS[i].pos[0]+timeoffset) * (this->scalarsAOS[i].pos[0]+timeoffset) +
             (this->scalarsAOS[i].pos[1]+timeoffset) * (this->scalarsAOS[i].pos[1]+timeoffset) +
             (this->scalarsAOS[i].pos[2]+timeoffset) * (this->scalarsAOS[i].pos[2]+timeoffset));
      this->scalarsAOS[i].rho = R;
      this->scalarsAOS[i].temp = sqrt(R);
      this->scalarsAOS[i].vel[0] = R;
      this->scalarsAOS[i].vel[1] = R;
      this->scalarsAOS[i].vel[2] = R;
#else
      T R = ((this->x[i]+timeoffset)*(this->x[i]+timeoffset) + /* rho is equal to radius_square */
             (this->y[i]+timeoffset)*(this->y[i]+timeoffset) +
             (this->z[i]+timeoffset)*(this->z[i]+timeoffset));
      this->rho[i] = R;
      this->temp[i] = sqrt(R);
      this->vx[i] = R;
      this->vy[i] = R;
      this->vz[i] = R;
#endif
      }

#ifndef STRIDED_SCALARS
    auto minmax = std::minmax_element(this->rho.begin(), this->rho.end());
#endif
    };
};
}
#endif
