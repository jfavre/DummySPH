#pragma once

#include <iostream>
#include <string>
#include "solvers.h"

#ifdef USE_CATALYST
#include <catalyst.hpp>
//#include <catalyst_conduit.h>
#include <conduit_blueprint.h>
typedef conduit_cpp::Node ConduitNode;
#endif

#ifdef USE_ASCENT
#include <ascent/ascent.hpp>
#include <cuda_helpers.cpp>
typedef conduit::Node ConduitNode;
#endif

#if defined(USE_CATALYST) || defined(USE_ASCENT)
template<typename T>
void addField(ConduitNode& mesh, const std::string& name, T* field, const size_t N)
{
    mesh["fields/" + name + "/association"] = "vertex";
    mesh["fields/" + name + "/topology"]    = "mesh";
    mesh["fields/" + name + "/values"].set_external(field, N);
    mesh["fields/" + name + "/volume_dependent"].set("false");
}

template<typename T>
void addStridedField(ConduitNode& mesh,
                     const std::string& name,
                     T* field,
                     const size_t N,  // num_elements
                     const int offset,
                     const int stride)
{
    mesh["fields/" + name + "/association"] = "vertex";
    mesh["fields/" + name + "/topology"]    = "mesh";
    mesh["fields/" + name + "/values"].set_external(field, N,
                                                    offset * sizeof(T),
                                                    stride * sizeof(T));
    mesh["fields/" + name + "/volume_dependent"].set("false");
}

template<typename T>
void addCoordinates(ConduitNode& mesh, std::vector<T> &x, std::vector<T> &y, std::vector<T> &z)
{
  mesh["coordsets/coords/values/x"].set_external(x);
  mesh["coordsets/coords/values/y"].set_external(y);
  mesh["coordsets/coords/values/z"].set_external(z);
}

template<typename T>
void addStridedCoordinates(ConduitNode& mesh,
                           T* xyz,
                           const size_t N,  /* num_elements */
                           const int stride)
{
  mesh["coordsets/coords/values/x"].set_external(xyz, N, 0 * sizeof(T), stride * sizeof(T));
  mesh["coordsets/coords/values/y"].set_external(xyz, N, 1 * sizeof(T), stride * sizeof(T));
  mesh["coordsets/coords/values/z"].set_external(xyz, N, 2 * sizeof(T), stride * sizeof(T));
}
#endif
  
#ifdef USE_CATALYST
#include "CatalystAdaptor.h"
#endif

#ifdef USE_ASCENT
#include "AscentAdaptor.h"
#endif

#ifdef USE_VTKM
#include "VTKmAdaptor.h"
#endif

#include "timer.hpp"



namespace viz
{

template<typename T>
void init([[maybe_unused]] int argc,
               [[maybe_unused]] char** argv,
               [[maybe_unused]] sph::ParticlesData<T> *sim)
{
#ifdef USE_CATALYST
  CatalystAdaptor::Initialize(argc, argv);
  std::cout << "CatalystInitialize\n";
#endif
#ifdef USE_ASCENT
  AscentAdaptor::Initialize(sim);
  std::cout << "AscentInitialize\n";
#endif
#ifdef USE_VTKM
  VTKmAdaptor::Initialize(argc, argv, sim);
  std::cout << "VTK-mInitialize\n";
#endif
}

template<typename T>
void execute([[maybe_unused]] sph::ParticlesData<T> *sim,
             [[maybe_unused]] int iteration,
             [[maybe_unused]] int frequency,
             const std::string &testname, const std::string &FileName)
{
#ifdef USE_CATALYST
  CatalystAdaptor::Execute(sim);
#endif
#ifdef USE_ASCENT
  AscentAdaptor::Execute(iteration, frequency, sim, testname, FileName);
#endif
#ifdef USE_VTKM
  VTKmAdaptor::Execute(iteration, frequency, sim->par_rank, testname, FileName);
#endif
}

void finalize([[maybe_unused]] int &rank)
{
#ifdef USE_CATALYST
  CatalystAdaptor::Finalize();
#endif
#ifdef USE_ASCENT
  std::cout << "AscentFinalize\n";
  AscentAdaptor::Finalize();
#endif
#ifdef USE_VTKM
  std::cout << "VTK-mFinalize\n";
  VTKmAdaptor::Finalize(rank);
#endif
}

} // namespace viz
