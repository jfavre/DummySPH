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

using namespace std;

namespace AscentAdaptor
{
  ascent::Ascent ascent;
  ConduitNode  mesh;
  ConduitNode  actions;

template<typename T>
void Initialize(sph::ParticlesData<T> *sim,
                int frequency,
                const string &testname,
                const string &FileName)
{
  ConduitNode n;
  ascent::about(n);
  // only run this test if ascent was built with vtkm support
  if(n["runtimes/ascent/vtkm/status"].as_string() == "disabled")
  {
    ASCENT_INFO("Ascent vtkm support disabled, skipping test");
    return;
  }

  string output_path = "datasets/";
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
  ascent_options["runtime/vtkm/backend"] = "cuda";
#endif
  ascent.open(ascent_options);

  mesh["state/cycle"].set_external(&sim->iteration);
  mesh["state/time"].set_external(&sim->time);
  mesh["state/domain_id"].set_external(&sim->par_rank);

  std::cout << "time: " << sim->iteration*0.1 << " cycle: " << sim->iteration << std::endl;

  mesh["coordsets/coords/type"] = "explicit";
  mesh["topologies/mesh/coordset"] = "coords";
  
//#define IMPLICIT_CONNECTIVITY_LIST 1 // the connectivity list is not given, but created by vtkm
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

  //
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
    int data_nbytes = sim->n * sizeof(T);
    
  device_move(mesh["coordsets/coords/values/x"],             data_nbytes);
  device_move(mesh["coordsets/coords/values/y"],             data_nbytes);
  device_move(mesh["coordsets/coords/values/z"],             data_nbytes);
  device_move(mesh["topologies/mesh/elements/connectivity"], sim->n*sizeof(conduit_int32));
  device_move(mesh["fields/rho/values"],                     data_nbytes);
  device_move(mesh["fields/Temperature/values"],             data_nbytes);
  device_move(mesh["fields/mass/values"],                    data_nbytes);
  device_move(mesh["fields/vx/values"],                      data_nbytes);
  device_move(mesh["fields/vy/values"],                      data_nbytes);
  device_move(mesh["fields/vz/values"],                      data_nbytes);
  device_move(mesh["fields/x/values"],                       data_nbytes);
  device_move(mesh["fields/y/values"],                       data_nbytes);
  device_move(mesh["fields/z/values"],                       data_nbytes);
#endif
#endif
  conduit::Node verify_info;
  if(!conduit::blueprint::mesh::verify(mesh,verify_info))
  {
    CONDUIT_INFO("blueprint verify failed!" + verify_info.to_json());
  }
  if(conduit::blueprint::mcarray::is_interleaved(mesh["coordsets/coords/values"]))
    std::cout << "Conduit Blueprint check found interleaved coordinates" << std::endl;
  else
    std::cout << "Conduit Blueprint check found contiguous coordinates" << std::endl;
  //mesh.print();

  string trigger_file = conduit::utils::join_file_path("./","simple_trigger_actions.yaml");
  conduit::utils::remove_path_if_exists(trigger_file);

  ConduitNode &add_triggers = actions.append();
  add_triggers["action"] = "add_triggers";
  ConduitNode &triggers = add_triggers["triggers"];
  triggers["t1/params/condition"] = "cycle() % " + std::to_string(frequency) + " == 0";
  triggers["t1/params/actions_file"] = "simple_trigger_actions.yaml";

    ConduitNode trigger_actions;
    if (!testname.compare("dumping"))
      {
      ConduitNode extracts;
      extracts["e1/type"]  = "relay";
      extracts["e1/params/path"] = output_path + FileName.c_str();
      extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
      extracts["e1/params/fields"].append() = "rho";

      ConduitNode &add_ext= trigger_actions.append();
      add_ext["action"] = "add_extracts";
      add_ext["extracts"] = extracts;
      }
    else if (!testname.compare("histsampling"))
      {
      ConduitNode pipelines;
      pipelines["p1/f1/type"]  = "histsampling";
      pipelines["p1/f1/params/sample_rate"] = 0.05;
      pipelines["p1/f1/params/bins"] = 64;
      pipelines["p1/f1/params/field"] = "rho";

      ConduitNode extracts;
      extracts["e1/type"]  = "relay";
      extracts["e1/pipeline"] = "p1";
      extracts["e1/params/path"] = output_path + FileName.c_str();
      extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
      extracts["e1/params/fields"].append() = "rho";

      ConduitNode &add_pip = trigger_actions.append();
      add_pip["action"] = "add_pipelines";
      add_pip["pipelines"] = pipelines;

      ConduitNode &add_ext= trigger_actions.append();
      add_ext["action"] = "add_extracts";
      add_ext["extracts"] = extracts;
      }
    else if (!testname.compare("thresholding"))
      {
      ConduitNode queries;
      queries["q1/params/expression"] = "min(field('rho')).value";
      queries["q1/params/name"] = "min_rho";
      queries["q2/params/expression"] = "max(field('rho')).value";
      queries["q2/params/name"] = "max_rho";

      ConduitNode &add_query = trigger_actions.append();
      add_query["action"] = "add_queries";
      add_query["queries"] = queries;
      
      ConduitNode pipelines;
 
      pipelines["p1/f1/type"]  = "threshold";
      pipelines["p1/f1/params/field"] = "rho";
      pipelines["p1/f1/params/min_value"] = "min_rho + 0.25 * (max_rho - min_rho)";
      pipelines["p1/f1/params/max_value"] = "max_rho - 0.25 * (max_rho - min_rho)";

      ConduitNode &add_pip = trigger_actions.append();
      add_pip["action"] = "add_pipelines";
      add_pip["pipelines"] = pipelines;

      ConduitNode extracts;
      extracts["e1/type"]  = "relay";
      extracts["e1/pipeline"] = "p1";
      extracts["e1/params/path"] = output_path + FileName.c_str();
      extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
      extracts["e1/params/fields"].append() = "rho";

      ConduitNode &add_ext= trigger_actions.append();
      add_ext["action"] = "add_extracts";
      add_ext["extracts"] = extracts;
      }
    else if (!testname.compare("compositing"))
      {
      ConduitNode pipelines;
      pipelines["p1/f1/type"]  = "composite_vector";
      pipelines["p1/f1/params/field1"] = "vx";
      pipelines["p1/f1/params/field2"] = "vy";
      pipelines["p1/f1/params/field3"] = "vz";
      //don't call it "velocity" to avoid a potential clash with an already defined "velocity"
      pipelines["p1/f1/params/output_name"] = "vxvyvz";
// it seems like vector fields are not supported to be written as Blueprint HDF5
// "Field type unsupported for conversion to blueprint."
// so I add a vector magnitude operator
      pipelines["p1/f2/type"]  = "vector_magnitude";
      pipelines["p1/f2/params/field"] = "vxvyvz";
      pipelines["p1/f2/params/output_name"] = "velocity_magnitude";

      ConduitNode extracts;
      extracts["e1/type"]  = "relay";
      extracts["e1/pipeline"] = "p1";
      extracts["e1/params/path"] = output_path + FileName.c_str();
      extracts["e1/params/protocol"] = "blueprint/mesh/hdf5";
      extracts["e1/params/fields"].append() = "rho";
      extracts["e1/params/fields"].append() = "velocity_magnitude";

      ConduitNode &add_pip = trigger_actions.append();
      add_pip["action"] = "add_pipelines";
      add_pip["pipelines"] = pipelines;

      ConduitNode &add_ext= trigger_actions.append();
      add_ext["action"] = "add_extracts";
      add_ext["extracts"] = extracts;
      }
    else if (!testname.compare("rendering"))
      {
      ConduitNode scenes;
      scenes["s1/plots/p1/type"] = "pseudocolor";
      scenes["s1/plots/p1/field"] = "rho";
      scenes["s1/plots/p1/color_table/name"] = "viridis";
      scenes["s1/renders/r1/color_bar_position"].set({-0.9,0.9,0.8,0.85});
      scenes["s1/renders/r1/camera/azimuth"] = 30.0;
      scenes["s1/renders/r1/camera/elevation"] = 30.0;
      scenes["s1/renders/r1/image_prefix"] = output_path + FileName + "_%04d"; 
      scenes["s1/renders/r1/bg_color"].set({1.,1.,1.});
      scenes["s1/renders/r1/fg_color"].set({0.,0.,0.});
      ConduitNode &add_scene = trigger_actions.append();
      add_scene["action"] = "add_scenes";
      add_scene["scenes"] = scenes;
      }
    else if (!testname.compare("binning"))
      {
      // in this particular case, we use FileName to hold the variable name, e.g. rho
      ConduitNode queries;
      queries["q1/params/expression"] = "min(field('" + FileName + "'))";
      queries["q1/params/name"] = "min_" + FileName;
      queries["q2/params/expression"] = "max(field('" + FileName + "'))";
      queries["q2/params/name"] = "max_" + FileName;

      ConduitNode &add_query = trigger_actions.append();
      add_query["action"] = "add_queries";
      add_query["queries"] = queries;
      std::cout << "See the output in datasets/ascent_session.yaml" << std::endl;
      }
    trigger_actions.save(trigger_file);

    //std::cout << trigger_actions.to_yaml() << std::endl;
    //std::cout << actions.to_yaml() << std::endl;
}

template<typename T>
void Execute([[maybe_unused]]int it, sph::ParticlesData<T> *sim)
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
