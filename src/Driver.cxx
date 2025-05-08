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

// include the option parser from https://optionparser.sourceforge.net/
#include "optionparser.h"

enum  optionIndex { UNKNOWN, HELP, TIPSY, H5PART, NPARTICLES, RENDERING, COMPOSITING, THRESHOLDING, HISTSAMPLING, DUMPING, BINNING, CATALYST};
const option::Descriptor usage[] =
{
 {UNKNOWN, 0,"" , ""    ,option::Arg::None, "USAGE: dummysph_* [options]\n\n"
                                            "Options:" },
 {HELP,            0,"" , "help",option::Arg::None, "  --help  \tPrint usage and exit." },
 #ifdef LOAD_TIPSY
 {TIPSY,           0,"tipsy" , "tipsy",option::Arg::Required, "  --tipsy <filename> \t(reads a PKDGRAV3 dump)" },
 #endif
 #ifdef LOAD_H5Part
 {H5PART,          0,"h5part" , "h5part",option::Arg::Required, "  --h5part <filename> \t(reads an SPH-EXA dump)" },
 #endif
 {NPARTICLES,      0,"n", "n",option::Arg::Numeric, "  --n <num> \tNumber of particles" },
 #ifdef USE_ASCENT
 {RENDERING,       0,"rendering" , "rendering",option::Arg::Required, "  --rendering  <filename> \t(makes a PNG file)" },
 {COMPOSITING,     0,"compositing" , "compositing",option::Arg::Required, "  --compositing <filename> \t(dumps a Conduit Blueprint HDF5 file)" },
 {THRESHOLDING,    0,"thresholding" , "thresholding",option::Arg::Required, "  --thresholding <filename> \t(dumps a Conduit Blueprint HDF5 file)" },
 {HISTSAMPLING,    0,"histsampling" , "histsampling",option::Arg::Required, "  --histsampling <filename> \t(dumps a Conduit Blueprint HDF5 file)" },
 {DUMPING,         0,"dumping" , "dumping",option::Arg::Required, "  --dumping <filename> \t(dumps a Conduit Blueprint HDF5 or VTK file)" },
 {BINNING,         0,"binning" , "binning",option::Arg::Required, "  --binning <varname>\t(results are in ascent_session.yaml file)" },
#endif
#ifdef USE_CATALYST
 {CATALYST,    0,"catalyst" , "catalyst",option::Arg::Required, "  --catalyst <filename.py> \t(executes a ParaView Catalyst Python script)" },
#endif
 {UNKNOWN, 0,"" ,  ""   ,option::Arg::Required, "\nExamples:\n"
                                            "  dummysph_* --compositing filename\n"
                                            "  dummysph_* --n 100 --rendering filename\n"
                                            "  dummysph_* --tipsy hr8799_bol_bd1.017300\n"
                                            },
 {0,0,0,0,0,0}
};

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

  argc -= (argc>0); argv+=(argc>0); // skip program name argv[0] if present
  option::Stats  stats(usage, argc, argv);
  option::Option options[stats.options_max], buffer[stats.buffer_max];
  option::Parser parse(usage, argc, argv, options, buffer);
  if (parse.error())
    return 1;
  if (options[HELP] || argc == 0) {
    option::printUsage(std::cout, usage);
    return 0;
  }

  for (int i = 0; i < parse.optionsCount(); ++i)
  {
    option::Option& opt = buffer[i];
    switch (opt.index())
    {
      case HELP:
        // not possible, because handled further above and exits the program
      case NPARTICLES:
        Nparticles = atoi(opt.arg);
        break;
      case TIPSY:
        TipsyFileName = opt.arg;
        dummydata = false;
        break;
      case H5PART:
        H5PartFileName = opt.arg;
        dummydata = false;
        break;
     case HISTSAMPLING:
        FileName = opt.arg;
        testname = "histsampling";
        break;
     case RENDERING:
        FileName = opt.arg;
        testname = "rendering";
        break;
     case THRESHOLDING:
        FileName = opt.arg;
        testname = "thresholding";
        break;
     case COMPOSITING:
        FileName = opt.arg;
        testname = "compositing";
        break;
     case BINNING:
        FileName = opt.arg;
        testname = "binning";
        break;
     case DUMPING:
        FileName = opt.arg;
        testname = "dumping";
        break;
     case CATALYST:
        FileName = opt.arg;
        testname = "catalyst";
        break;
     case UNKNOWN:
        std::cerr << "Unknown option\n"; option::printUsage(std::cerr, usage); exit(1);
        break;
    }
  }

  for (int i = 0; i < parse.nonOptionsCount(); ++i)
    std::cout << "Non-option #" << i << ": " << parse.nonOption(i) << "\n";
    
#ifdef SPH_DOUBLE
  ParticlesData<double> *sim = new(ParticlesData<double>);
#else
  ParticlesData<float> *sim = new(ParticlesData<float>);
#endif

  if(dummydata)
    sim->AllocateGridMemory(Nparticles);

#ifdef LOAD_TIPSY
    // only knows how to load a static Tipsy file at the moment.
    int n[3] = {1,0,0};
    frequency = Niterations = 1;
    TipsyFile *filein = new TipsyFile(TipsyFileName.c_str());
    filein->read_header();
    filein->read_gas_piece(par_rank, par_size, n[0]);
    sim->UseTipsyData(filein->gas_ptr(), n[0]);
    delete filein;
#endif

#ifdef LOAD_H5Part
    // only knows how to load a single timestep at the moment
    frequency = Niterations = 1;
    sim->UseH5PartData(H5PartFileName);
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

