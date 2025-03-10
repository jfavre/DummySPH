#ifndef AscentAdaptor_h
#define AscentAdaptor_h

/*
publish() leads to following call path

VTKHDataAdapter::BlueprintToVTKHCollection()
VTKHDataAdapter::PointsImplicitBlueprintToVTKmDataSet()
        coords = detail::GetExplicitCoordinateSystem<float64>() which
        allocates connectivity.Allocate(nverts);
*/

#include "conduit_blueprint.hpp"
#ifdef CAMP_HAVE_CUDA
#include "cuda_helpers.cpp"
#endif
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>

namespace AscentAdaptor
{
  ascent::Ascent ascent;
  ConduitNode  mesh;
  ConduitNode  actions;

template<typename T>
void Initialize(sph::ParticlesData<T> *sim)
{
  ConduitNode n;
  ascent::about(n);
  // only run this test if ascent was built with vtkm support
  if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent vtkm support disabled, skipping test");
    return;
  }

  std::string output_path = "datasets";
  ASCENT_INFO("Creating output folder: " + output_path);
  if(!conduit::utils::is_directory(output_path))
  {
    conduit::utils::create_directory(output_path);
  }

  ConduitNode ascent_options;
  ascent_options["default_dir"] = output_path;
  ascent_options["mpi_comm"] = MPI_Comm_c2f(MPI_COMM_WORLD);
  ascent_options["ascent_info"] = "verbose";
  ascent_options["exceptions"] = "forward";
#ifdef CAMP_HAVE_CUDA
  ascent_options["runtine/vtkm/backend"] = "cuda";
#endif
  ascent.open(ascent_options);

  mesh["state/cycle"].set_external(&sim->iteration);
  mesh["state/time"].set_external(&sim->time);
  mesh["state/domain_id"].set_external(&sim->par_rank);
  conduit::Node verify_info;

  std::cout << "time: " << sim->iteration*0.1 << " cycle: " << sim->iteration << std::endl;

  mesh["coordsets/coords/type"] = "explicit";
  mesh["topologies/mesh/coordset"] = "coords";
  
#define IMPLICIT_CONNECTIVITY_LIST 1 // the connectivity list is not given, but created by vtkm
#ifdef  IMPLICIT_CONNECTIVITY_LIST
  mesh["topologies/mesh/type"] = "points";
#else
  mesh["topologies/mesh/type"] = "unstructured";
  std::vector<conduit_int32> conn(sim->n);
  std::iota(conn.begin(), conn.end(), 0);
  mesh["topologies/mesh/elements/connectivity"].set(conn);
  mesh["topologies/mesh/elements/shape"] = "point";
#endif

#ifdef STRIDED_SCALARS
  // first the coordinates
  addStridedCoordinates(mesh, &sim->scalarsAOS[0].pos[0], sim->n, sim->NbofScalarfields);
  
  // then the variables
  addStridedField(mesh, "rho",         &sim->scalarsAOS[0].rho,      sim->n, 0, sim->NbofScalarfields);
  addStridedField(mesh, "Temperature", &sim->scalarsAOS[0].temp,     sim->n, 0, sim->NbofScalarfields);
  addStridedField(mesh, "mass",        &sim->scalarsAOS[0].mass,     sim->n, 0, sim->NbofScalarfields);

  addStridedField(mesh, "x",           &(sim->scalarsAOS[0].pos[0]), sim->n, 0, sim->NbofScalarfields);
  addStridedField(mesh, "y",           &(sim->scalarsAOS[0].pos[1]), sim->n, 0, sim->NbofScalarfields);
  addStridedField(mesh, "z",           &(sim->scalarsAOS[0].pos[2]), sim->n, 0, sim->NbofScalarfields);

  addStridedField(mesh, "vx",          &(sim->scalarsAOS[0].vel[0]), sim->n, 0, sim->NbofScalarfields);
  addStridedField(mesh, "vy",          &(sim->scalarsAOS[0].vel[1]), sim->n, 0, sim->NbofScalarfields);
  addStridedField(mesh, "vz",          &(sim->scalarsAOS[0].vel[2]), sim->n, 0, sim->NbofScalarfields);

  /* there seems to be an offset I don't understand
  mesh["fields/velocity/association"] = "vertex";
  mesh["fields/velocity/topology"]    = "mesh";
  mesh["fields/velocity/values/u"].set_external(&(sim->scalarsAOS[0].vel[0]), sim->n,
                                                                0,
                                                                sim->NbofScalarfields * sizeof(T));
  mesh["fields/velocity/values/v"].set_external(&(sim->scalarsAOS[0].vel[1]), sim->n,
                                                                0 * sizeof(T),
                                                                sim->NbofScalarfields * sizeof(T));
  mesh["fields/velocity/values/w"].set_external(&(sim->scalarsAOS[0].vel[2]), sim->n,
                                                                0 * sizeof(T),
                                                                sim->NbofScalarfields * sizeof(T));
  mesh["fields/velocity/volume_dependent"].set("false");
  */
  
#else
  // first the coordinates
  addCoordinates(mesh, sim->x, sim->y, sim->z);
  
  // then the variables
  addField(mesh, "rho" , sim->rho.data(), sim->n);
  addField(mesh, "Temperature", sim->temp.data(), sim->n);
  addField(mesh, "mass", sim->mass.data(), sim->n);

  addField(mesh, "vx", sim->vx.data(), sim->n);
  addField(mesh, "vy", sim->vy.data(), sim->n);
  addField(mesh, "vz", sim->vz.data(), sim->n);
  
  addField(mesh, "x", sim->x.data(), sim->n);
  addField(mesh, "y", sim->y.data(), sim->n);
  addField(mesh, "z", sim->z.data(), sim->n);

  /*
  mesh["fields/velocity/association"] = "vertex";
  mesh["fields/velocity/topology"]    = "mesh";
  mesh["fields/velocity/values/u"].set_external(sim->vx.data(), sim->n);
  mesh["fields/velocity/values/v"].set_external(sim->vy.data(), sim->n);
  mesh["fields/velocity/values/w"].set_external(sim->vz.data(), sim->n);
  mesh["fields/velocity/volume_dependent"].set("false");
  */
#endif

#if defined (ASCENT_CUDA_ENABLED)
#ifdef STRIDED_SCALARS
    // Future work
#else
// device_move allocates and uses set external to provide data on the device
    device_move(mesh["coordsets/coords/values/x"], sim->n*sizeof(T));
    device_move(mesh["coordsets/coords/values/y"], sim->n*sizeof(T));
    device_move(mesh["coordsets/coords/values/z"], sim->n*sizeof(T));
    //device_move(mesh["topologies/mesh/elements/connectivity"], sim->n);
    device_move(mesh["fields/rho/values"], sim->n*sizeof(T));
    device_move(mesh["fields/Temperature/values"], sim->n*sizeof(T));
    device_move(mesh["fields/mass/values"], sim->n*sizeof(T));
    device_move(mesh["fields/vx/values"], sim->n*sizeof(T));
    device_move(mesh["fields/vy/values"], sim->n*sizeof(T));
    device_move(mesh["fields/vz/values"], sim->n*sizeof(T));
    device_move(mesh["fields/x/values"], sim->n*sizeof(T));
    device_move(mesh["fields/y/values"], sim->n*sizeof(T));
    device_move(mesh["fields/z/values"], sim->n*sizeof(T));
#endif
#endif

  if(!conduit::blueprint::mesh::verify(mesh,verify_info))
  {
    CONDUIT_INFO("blueprint verify failed!" + verify_info.to_json());
  }
  if(conduit::blueprint::mcarray::is_interleaved(mesh["coordsets/coords/values"]))
    std::cout << "Conduit Blueprint check found interleaved coordinates" << std::endl;
  else
    std::cout << "Conduit Blueprint check found contiguous coordinates" << std::endl;
  //mesh.print();
// Create an action that tells Ascent to:
//  add a scene (s1) with one plot (p1)
//  that will render a pseudocolor of 
//  the mesh field `rho`

  ConduitNode &add_act = actions.append(); 
  add_act["action"] = "add_scenes";

// declare a scene (s1) and pseudocolor plot (p1)
  ConduitNode &scenes = add_act["scenes"];
  scenes["s1/plots/p1/type"] = "pseudocolor";
//#define RANKS
#ifdef RANKS
  scenes["s1/plots/p1/field"] = "ranks";
  scenes["s1/plots/p1/pipeline"] = "pl1";
  scenes["s1/plots/p1/color_table/discrete"] = "true";
#else
  scenes["s1/plots/p1/field"] = "rho";
#endif
  //scenes["s1/plots/p1/points/radius"] = .005;
  //scenes["s1/plots/p1/points/radius_delta"] = .01;
  scenes["s1/renders/r1/image_prefix"] = "image.%05d";
  scenes["s1/renders/r1/annotations"] = "true";
  double vec3[3];
  vec3[0] = (sim->par_size-1.0)/2; vec3[1] = (sim->par_size-1.0)/2; vec3[2] = 0.0;
  scenes["s1/renders/r1/camera/look_at"].set_float64_ptr(vec3,3);
  vec3[0] = (sim->par_size-1.0)/2; vec3[1] = (sim->par_size-1.0)/2; vec3[2] = 10;
  scenes["s1/renders/r1/camera/position"].set_float64_ptr(vec3,3);
  vec3[0] = 0.0; vec3[1] = 1; vec3[2] = 0.0;
  scenes["s1/renders/r1/camera/up"].set_float64_ptr(vec3,3);
  scenes["s1/renders/r1/camera/zoom"] = 5.0/sim->par_size;
  scenes["s1/renders/r1/image_width"] = 1024;
  scenes["s1/renders/r1/image_height"] = 1024;
  double dset_bounds[6] = {-1.0, 1.0 * sim->par_size, -1.0, 1.0 * sim->par_size, 0.0, 1.};
  scenes["s1/renders/r1/dataset_bounds"].set_float64_ptr(dset_bounds, 6);
/*
  conduit::Node pipelines;
  pipelines["pl1/f1/type"] = "add_mpi_ranks";
  conduit::Node &params = pipelines["pl1/f1/params"];
  params["topology"] = "mesh";
  params["output"] = "ranks";

  conduit::Node &add_pipelines = actions.append();
  add_pipelines["action"] = "add_pipelines";
  add_pipelines["pipelines"] = pipelines;
  */
}

template<typename T>
void Execute([[maybe_unused]]int it, [[maybe_unused]]int frequency, sph::ParticlesData<T> *sim)
{
  if(it % frequency == 0)
    {
#if defined (ASCENT_CUDA_ENABLED)
#ifdef STRIDED_SCALARS
    // Future work
#else
    // update "rho" and "temp" on device
    copy_from_host_to_device(mesh["fields/rho/values"].data_ptr(),
                             sim->rho.data(), sim->n*sizeof(T));
    copy_from_host_to_device(mesh["fields/Temperature/values"].data_ptr(),
                             sim->temp.data(), sim->n*sizeof(T));
    copy_from_host_to_device(mesh["fields/vx/values"].data_ptr(),
                             sim->vx.data(), sim->n*sizeof(T));
    copy_from_host_to_device(mesh["fields/vy/values"].data_ptr(),
                             sim->vy.data(), sim->n*sizeof(T));
    copy_from_host_to_device(mesh["fields/vz/values"].data_ptr(),
                             sim->vz.data(), sim->n*sizeof(T));
#endif
#endif
    ascent.publish(mesh);
    ascent.execute(actions);
    }
}

//#define DATADUMP 1
void Finalize()
{
#ifdef DATADUMP
  ConduitNode save_data_actions;
  ConduitNode &add_act = save_data_actions.append();
  add_act["action"] = "add_extracts";
  conduit::Node &extracts = add_act["extracts"];
  extracts["e1/type"] = "relay";
  extracts["e1/params/path"] = "mesh";
  extracts["e1/params/protocol"] = "hdf5";

  ascent.publish(mesh);
  ascent.execute(save_data_actions);
  ascent.close();
#endif
}

}
#endif
