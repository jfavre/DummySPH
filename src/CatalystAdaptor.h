#ifndef CatalystAdaptor_h
#define CatalystAdaptor_h

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>

namespace CatalystAdaptor
{
/**
 * In this example, we show how we can use Catalysts's C++
 * wrapper around conduit's C API to create Conduit nodes.
 * This is not required. A C++ adaptor can just as
 * conveniently use the Conduit C API to setup the
 * `conduit_node`. However, this example shows that one can
 * indeed use Catalyst's C++ API, if the developer so chooses.
 */
void Initialize(int argc, char* argv[])
{
  ConduitNode node;
  for (int cc = 1; cc < argc; ++cc)
  {
    if(strcmp(argv[cc], "--help") == 0 || strcmp(argv[cc], "-h") == 0)
    {
      std::cout << "Runtime Options: " << std::endl;
      std::cout << "               : --help or -h (help)" << std::endl;
      std::cout << "               : --catalyst script.py" << std::endl;
      std::cout << "               : --pv (choose the ParaView Catalyst)" << std::endl;
      exit(1);
    }
    else if (strcmp(argv[cc], "--output") == 0 && (cc + 1) < argc)
    {
      node["catalyst/pipelines/0/type"].set("io");
      node["catalyst/pipelines/0/filename"].set(argv[cc + 1]);
      node["catalyst/pipelines/0/channel"].set("grid");
      ++cc;
    }
    else if (strcmp(argv[cc], "--pv") == 0)
    {
      node["catalyst_load/implementation"].set_string("paraview");
      //std::cout << "Loading ParaView implementation\n";
    }
    else if (strcmp(argv[cc], "--catalyst") == 0 && (cc + 1) < argc)
    {
      const auto fname = std::string(argv[cc+1]);
      // note: one can simply add the script file as follows:
      // node["catalyst/scripts/script" + std::to_string(cc - 1)].set_string(path);

      // alternatively, use this form to pass optional parameters to the script.
      const auto path = "catalyst/scripts/script" + std::to_string(cc - 1);
      node[path + "/filename"].set_string(fname);
    }
  }

  catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
  if (err != catalyst_status_ok)
    std::cerr << "ERROR: Failed to initialize Catalyst: " << err << std::endl;
}



//#define IMPLICIT_CONNECTIVITY_LIST 1

template<typename T>
void Execute(sph::ParticlesData<T> *sim)
{
  ConduitNode exec_params;

  auto state = exec_params["catalyst/state"];
  state["timestep"].set(sim->iteration);
  state["time"].set(sim->iteration*0.1);

  // Add channels.
  // We only have 1 channel here. Let's name it 'grid'.
  auto channel = exec_params["catalyst/channels/grid"];

  // Since this example is using Conduit Mesh Blueprint to define the mesh,
  // we set the channel's type to "mesh".
  channel["type"].set("mesh");

  // now create the mesh.
  auto mesh = channel["data"];

  mesh["coordsets/coords/type"] = "explicit";
  mesh["topologies/mesh/coordset"] = "coords";
#ifdef STRIDED_SCALARS
  addStridedCoordinates(mesh, &sim->scalarsAOS[0].pos[0], sim->n, sim->NbofScalarfields);
#else
  addCoordinates(mesh, sim->x, sim->y, sim->z);
#endif

#ifdef IMPLICIT_CONNECTIVITY_LIST
  mesh["topologies/mesh/type"] = "points";
#else
  mesh["topologies/mesh/type"] = "unstructured";
  std::vector<conduit_int32> conn(sim->n);
  std::iota(conn.begin(), conn.end(), 0);
  mesh["topologies/mesh/elements/connectivity"].set(conn);
  mesh["topologies/mesh/elements/shape"] = "point";
#endif
  mesh["topologies/mesh/coordset"].set("coords");
  
#ifdef STRIDED_SCALARS

  addStridedField(mesh, "rho",  &sim->scalarsAOS[0].rho, sim->n, 0, sim->NbofScalarfields);
  addStridedField(mesh, "temp", &sim->scalarsAOS[0].temp, sim->n, 0, sim->NbofScalarfields);

  addStridedField(mesh, "x", &(sim->scalarsAOS[0].pos[0]), sim->n, 0, sim->NbofScalarfields);
  addStridedField(mesh, "y", &(sim->scalarsAOS[0].pos[0]), sim->n, 1, sim->NbofScalarfields);
  addStridedField(mesh, "z", &(sim->scalarsAOS[0].pos[0]), sim->n, 2, sim->NbofScalarfields);
  addStridedField(mesh, "vx", &(sim->scalarsAOS[0].vel[0]), sim->n, 0, sim->NbofScalarfields);
  addStridedField(mesh, "vy", &(sim->scalarsAOS[0].vel[0]), sim->n, 1, sim->NbofScalarfields);
  addStridedField(mesh, "vz", &(sim->scalarsAOS[0].vel[0]), sim->n, 2, sim->NbofScalarfields);
  /*
   there seems to be an offset I don't understand
  mesh["fields/velocity/association"] = "vertex";
  mesh["fields/velocity/topology"]    = "mesh";
  mesh["fields/velocity/values/u"].set_external_float32_ptr(&(sim->scalarsAOS[0].vel[0]), sim->n,
                                                                0,
                                                                sim->NbofScalarfields * sizeof(T));
  mesh["fields/velocity/values/v"].set_external_float32_ptr(&(sim->scalarsAOS[0].vel[0]), sim->n,
                                                /                1 * sizeof(T),
                                                                sim->NbofScalarfields * sizeof(T));
  mesh["fields/velocity/values/w"].set_external_float32_ptr(&(sim->scalarsAOS[0].vel[0]), sim->n,
                                                                2 * sizeof(T),
                                                                sim->NbofScalarfields * sizeof(T));
  mesh["fields/velocity/volume_dependent"].set("false");
  */

#else
  addField(mesh, "rho" , sim->rho.data(), sim->n);
  addField(mesh, "x" , sim->x.data(), sim->n);
  addField(mesh, "y" , sim->y.data(), sim->n);
  addField(mesh, "z" , sim->z.data(), sim->n);
  addField(mesh, "vx" , sim->vx.data(), sim->n);
  addField(mesh, "vy" , sim->vy.data(), sim->n);
  addField(mesh, "vz" , sim->vz.data(), sim->n);

  mesh["fields/velocity/association"] = "vertex";
  mesh["fields/velocity/topology"]    = "mesh";
  mesh["fields/velocity/values/u"].set_external(sim->vx.data(), sim->n);
  mesh["fields/velocity/values/v"].set_external(sim->vy.data(), sim->n);
  mesh["fields/velocity/values/w"].set_external(sim->vz.data(), sim->n);
  mesh["fields/velocity/volume_dependent"].set("false");
#endif

  ConduitNode verify_info;
  if (!conduit_blueprint_verify("mesh", conduit_cpp::c_node(&mesh), conduit_cpp::c_node(&verify_info)))
    std::cerr << "ERROR: blueprint verify failed!" + verify_info.to_json() << std::endl;
  //else std::cerr << "PASS: blueprint verify passed!"<< std::endl;

  catalyst_status err = catalyst_execute(conduit_cpp::c_node(&exec_params));
  if (err != catalyst_status_ok)
  {
    std::cerr << "ERROR: Failed to execute Catalyst: " << err << std::endl;
  }

  //conduit_node_save(c_node(&mesh), "my_output.json", "json");

}

void Finalize()
{
  ConduitNode node;
  catalyst_status err = catalyst_finalize(conduit_cpp::c_node(&node));
  if (err != catalyst_status_ok)
    std::cerr << "ERROR: Failed to finalize Catalyst: " << err << std::endl;
}
}

#endif
