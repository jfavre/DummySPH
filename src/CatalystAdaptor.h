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
void Initialize(int argc, char* argv[])
{
  ConduitNode node;
  for (int cc = 0; cc < argc; ++cc)
  {
    if (strcmp(argv[cc], "--catalyst") == 0 && (cc + 1) < argc)
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

//#define IMPLICIT_CONNECTIVITY_LIST 1
#ifdef IMPLICIT_CONNECTIVITY_LIST
  mesh["topologies/mesh/type"] = "points";
#else
  mesh["topologies/mesh/type"] = "unstructured";
  std::vector<conduit_int32> conn(sim->n);
  std::iota(conn.begin(), conn.end(), 0);
  mesh["topologies/mesh/elements/connectivity"].set_external(conn.data(), sim->n);
  mesh["topologies/mesh/elements/shape"] = "point";
#endif
  mesh["topologies/mesh/coordset"].set("coords");
  
#ifdef STRIDED_SCALARS
#ifdef MOVE_DATA_TO_DEVICE
  addStridedField_gpu(mesh, &sim->scalarsAOS[0].mass, sim->n, sim->NbofScalarfields);
#else
  addStridedCoordinates(mesh, &sim->scalarsAOS[0].pos[0], sim->n, sim->NbofScalarfields);
  
  addStridedField(mesh, "rho",  &sim->scalarsAOS[0].mass, sim->n, 7, sim->NbofScalarfields);
  addStridedField(mesh, "temp", &sim->scalarsAOS[0].mass, sim->n, 8, sim->NbofScalarfields);

  addStridedField(mesh, "x", &(sim->scalarsAOS[0].mass), sim->n, 1, sim->NbofScalarfields);
  addStridedField(mesh, "y", &(sim->scalarsAOS[0].mass), sim->n, 2, sim->NbofScalarfields);
  addStridedField(mesh, "z", &(sim->scalarsAOS[0].mass), sim->n, 3, sim->NbofScalarfields);
  addStridedField(mesh, "vx", &(sim->scalarsAOS[0].mass), sim->n, 4, sim->NbofScalarfields);
  addStridedField(mesh, "vy", &(sim->scalarsAOS[0].mass), sim->n, 5, sim->NbofScalarfields);
  addStridedField(mesh, "vz", &(sim->scalarsAOS[0].mass), sim->n, 6, sim->NbofScalarfields);
#endif
#else

#ifdef MOVE_DATA_TO_DEVICE
  std::cout << __FILE__ << ": " << __LINE__ << ": copying individual fields data to device\n";
// allocates new mem on device and uses set external to provide data on the device

#ifndef IMPLICIT_CONNECTIVITY_LIST
//device_move(mesh["topologies/mesh/elements/connectivity"], sim->n*sizeof(conduit_int32));
  int data_nbytes = sim->n*sizeof(conduit_int32);
  void *device_ptr = device_alloc(data_nbytes);
  if (device_ptr != nullptr)
    {
    copy_from_host_to_device(device_ptr, conn.data(), data_nbytes);
    mesh["topologies/mesh/elements/connectivity"].set_external(static_cast<conduit_int32*>(device_ptr), sim->n);
    conn.clear();
    }
  else
    std::cout << __FILE__ << ": " << __LINE__ << ":device-alloc returned null ptr\n";
#endif

  
  addField_gpu(mesh, "rho",         sim->rho.data(),  sim->n); sim->rho.clear();
  T* vxx = addField_gpu(mesh, "vx", sim->vx.data(),   sim->n); sim->vx.clear();
  T* vyy = addField_gpu(mesh, "vy", sim->vy.data(),   sim->n); sim->vy.clear();
  T* vzz = addField_gpu(mesh, "vz", sim->vz.data(),   sim->n); sim->vz.clear();
  addField_gpu(mesh, "Temperature", sim->temp.data(), sim->n); sim->temp.clear();
  addField_gpu(mesh, "mass",        sim->mass.data(), sim->n); sim->mass.clear();
  
  T* xx = addField_gpu(mesh, "x", sim->x.data(), sim->n); sim->x.clear();
  T* yy = addField_gpu(mesh, "y", sim->y.data(), sim->n); sim->y.clear();
  T* zz = addField_gpu(mesh, "z", sim->z.data(), sim->n); sim->z.clear();
  mesh["coordsets/coords/values/x"].set_external(xx, sim->n);
  mesh["coordsets/coords/values/y"].set_external(yy, sim->n);
  mesh["coordsets/coords/values/z"].set_external(zz, sim->n);
  
  mesh["fields/velocity/association"] = "vertex";
  mesh["fields/velocity/topology"]    = "mesh";
  mesh["fields/velocity/values/u"].set_external(vxx, sim->n);
  mesh["fields/velocity/values/v"].set_external(vyy, sim->n);
  mesh["fields/velocity/values/w"].set_external(vzz, sim->n);
  mesh["fields/velocity/volume_dependent"].set("false");
#else
  addCoordinates(mesh, sim->x, sim->y, sim->z);
  addField(mesh, "rho",         sim->rho.data(),  sim->n);
  addField(mesh, "x",           sim->x.data(),    sim->n);
  addField(mesh, "y",           sim->y.data(),    sim->n);
  addField(mesh, "z",           sim->z.data(),    sim->n);
  addField(mesh, "vx",          sim->vx.data(),   sim->n);
  addField(mesh, "vy",          sim->vy.data(),   sim->n);
  addField(mesh, "vz",          sim->vz.data(),   sim->n);
  addField(mesh, "Temperature", sim->temp.data(), sim->n);
  addField(mesh, "mass",        sim->mass.data(), sim->n);
  
  mesh["fields/velocity/association"] = "vertex";
  mesh["fields/velocity/topology"]    = "mesh";
  mesh["fields/velocity/values/u"].set_external(sim->vx.data(), sim->n);
  mesh["fields/velocity/values/v"].set_external(sim->vy.data(), sim->n);
  mesh["fields/velocity/values/w"].set_external(sim->vz.data(), sim->n);
  mesh["fields/velocity/volume_dependent"].set("false");
#endif // MOVE_DATA_TO_DEVICE

#endif // STRIDED_SCALARS

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
