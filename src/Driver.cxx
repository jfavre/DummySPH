/*
We currently have 3 different APIs for in-situ visualization

0) Catalyst
cd buildCatalyst2
make && ./bin/dummysph_catalystV2  --pv --catalyst ../catalyst_state.py

1) Ascent
cd buildAscent
make && ./bin/dummysph_ascent (# reads the ascent_actions.yaml present in current directory)

2) VTK-m
cd buildVTKm
make && ./bin/dummysph_vtkm


Written by Jean M. Favre, Swiss National Supercomputing Center
Tested Mon  9 Dec 13:50:37 CET 2024
*/

#include <mpi.h> 
#include <sstream>
#include <string>

#include "insitu_viz.h"
#ifdef LOAD_TIPSY
#include "tipsy_file.h"
#endif

using namespace sph;

int main(int argc, char *argv[])
{
  int it = 0, Niterations = 1, Nparticles = 100; // actually Nparticles^3
  int frequency = 1;
  int par_rank = 0;
  int par_size = 1;
  bool dummydata = true;
  const bool quiet = false;
  std::string TipsyFileName, H5PartFileName;
  std::string FileName, testname;
  std::ofstream nullOutput("/dev/null");
  std::ostream& output = (quiet || par_rank) ? nullOutput : std::cout;
  sphexa::Timer timer(output);
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &par_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &par_size);

  for (int cc = 1; cc < argc; ++cc)
  {
    if (strcmp(argv[cc], "--tipsy") == 0 && (cc + 1) < argc)
    {
      TipsyFileName = std::string(argv[cc+1]);
      dummydata = false;
    }
    if (strcmp(argv[cc], "--h5part") == 0 && (cc + 1) < argc)
    {
      H5PartFileName = std::string(argv[cc+1]);
      dummydata = false;
    }
    // now check for one test name
    if (strcmp(argv[cc], "--histsampling") == 0 && (cc + 1) < argc)
    {
      FileName = std::string(argv[cc+1]);
      testname = "histsampling";
    }
    if (strcmp(argv[cc], "--rendering") == 0 && (cc + 1) < argc)
    {
      FileName = std::string(argv[cc+1]);
      testname = "rendering";
    }
    if (strcmp(argv[cc], "--thresholding") == 0 && (cc + 1) < argc)
    {
      FileName = std::string(argv[cc+1]);
      testname = "thresholding";
    }
    if (strcmp(argv[cc], "--compositing") == 0 && (cc + 1) < argc)
    {
      FileName = std::string(argv[cc+1]);
      testname = "compositing";
    }
    if (strcmp(argv[cc], "--dumping") == 0 && (cc + 1) < argc)
    {
      FileName = std::string(argv[cc+1]);
      testname = "dumping";
    }
    if (strcmp(argv[cc], "--binning") == 0 && (cc + 1) < argc)
    {
      FileName = std::string(argv[cc+1]);
      testname = "binning";
    }
  }

#ifdef SPH_DOUBLE
  ParticlesData<double> *sim = new(ParticlesData<double>);
#else
  ParticlesData<float> *sim = new(ParticlesData<float>);
#endif
  if(dummydata)
    sim->AllocateGridMemory(Nparticles);
#ifdef LOAD_TIPSY
  else{
    // only knows how to load a static Tipsy file at the moment.
    int n[3] = {1,0,0};
    frequency = Niterations = 1;
    TipsyFile *filein = new TipsyFile(TipsyFileName.c_str());
    filein->read_header();
    filein->read_gas_piece(par_rank, par_size, n[0]);
    sim->UseTipsyData(filein->gas_ptr(), n[0]);
    delete filein;
  }
#endif
#ifdef LOAD_H5Part
  else{
    // only knows how to load a single timestep at the moment
    frequency = Niterations = 1;
    sim->UseH5PartData(H5PartFileName);
  }
#endif
  timer.start();
  timer.step("pre-initialization");
  
  viz::init(argc, argv, sim);
  
  timer.step("post-initialization");
  while (it < Niterations)
    {
    if(dummydata)
      sim->simulate_one_timestep();
    it++;
    viz::execute(sim, it, frequency, testname, FileName);
    }
  timer.step("post-exec");

  viz::finalize(par_rank);

  sim->FreeGridMemory();

  MPI_Finalize();

  return (0);
}

