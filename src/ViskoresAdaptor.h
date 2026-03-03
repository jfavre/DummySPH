#ifndef ViskoresAdaptor_h
#define ViskoresAdaptor_h

#include <iomanip>
#include <iostream>
#include <string>

#include <viskores/cont/ColorTable.h>
#include <viskores/cont/DataSet.h>
#include <viskores/cont/ArrayCopy.h>
#include <viskores/cont/ArrayCopyDevice.h>
#include <viskores/cont/DataSetBuilderExplicit.h>
#include <viskores/cont/ArrayHandleCompositeVector.h>
#include <viskores/cont/ArrayHandleStride.h>
#include <viskores/cont/ArrayHandleIndex.h>
#include <viskores/cont/Initialize.h>
#include <viskores/io/VTKDataSetWriter.h>
#include <viskores/rendering/Actor.h>
#include <viskores/rendering/CanvasRayTracer.h>
#include <viskores/filter/field_transform/LogValues.h>
#include <viskores/filter/field_transform/CompositeVectors.h>
#include <viskores/rendering/MapperPoint.h>
#include <viskores/filter/resampling/HistSampling.h>
#include <viskores/filter/geometry_refinement/ConvertToPointCloud.h>
#include <viskores/filter/entity_extraction/ThresholdPoints.h>
#include <viskores/rendering/Scene.h>
#include <viskores/rendering/View3D.h>
/****************************************

three methods have been identified to create coordinates array handles from the
given std::vector<T> x, y, z;         // Positions

1) viskores::cont::ArrayHandle<viskores::Vec3f> coordsArray0;
   coordsArray0.Allocate(static_cast<viskores::Id>(sim->n));

2) coordsArray1 = viskores::cont::make_ArrayHandleSOA<viskores::Vec3f>({sim->x, sim->y, sim->z});;

3) coordsArray2 = viskores::cont::make_ArrayHandleCompositeVector(component1, component2, component3);

****************************************/

namespace ViskoresAdaptor
{
  viskores::rendering::CanvasRayTracer canvas(1024,1024);
  viskores::rendering::Scene           scene;
  viskores::rendering::MapperPoint     mapper;
  viskores::cont::DataSet              dataSet;
  viskores::Bounds                     bounds;

void Execute_CompositeVectors(const std::string &filename);
void Execute_HistSampling(const std::string &filename);
void Execute_Rendering(const std::string &filename);
void Execute_ThresholdPoints(const std::string &filename);
void Execute_Dumping(const std::string &filename);

#ifdef Viskores_ENABLE_CUDA
template<typename T>
T *
device_alloc(int size, const T *data)
{
  void *buff;
  auto status = cudaMalloc(&buff, size * sizeof(T));
  if(status != cudaSuccess)
    {
    std::cerr << "error: CUDA API call : "
              << cudaGetErrorString(status) << std::endl;
    exit(1);
    }
  cudaMemcpy(buff, data, size * sizeof(T), cudaMemcpyHostToDevice);
  return static_cast<T*>(buff);
}
#endif

template<typename T>
void Initialize(int argc, char* argv[], sph::ParticlesData<T> *sim)
{
  std::cout << "Viskores::Initialize" << std::endl;

#ifdef Viskores_ENABLE_CUDA
  std::cout << "forcing DeviceAdapterTagCuda" << std::endl;
  viskores::cont::ScopedRuntimeDeviceTracker(viskores::cont::DeviceAdapterTagCuda{});
#endif

  viskores::cont::Initialize(argc, argv);

  viskores::cont::DataSetBuilderExplicit dataSetBuilder;
  /*
  viskores::cont::ArrayHandle<viskores::Vec3f> coordsArray0;
  coordsArray0.Allocate(static_cast<viskores::Id>(sim->n));

  auto coordsPortal = coordsArray0.WritePortal();
  for (std::size_t index = 0; index < sim->n; ++index)
  {
    coordsPortal.Set(static_cast<viskores::Id>(index),
                     viskores::make_Vec(static_cast<viskores::FloatDefault>(sim->x[index]),
                                    static_cast<viskores::FloatDefault>(sim->y[index]),
                                    static_cast<viskores::FloatDefault>(sim->z[index])));
  }
  std::cout << "COORDARRAY0 HANDLE" << std::endl;
  viskores::cont::printSummary_ArrayHandle(coordsArray0, std::cout);
  std::cout << "--------------------------" << std::endl;
  */
  /****************************************/
#ifdef STRIDED_SCALARS
  // NOTE: In this case, the num_vals, needs to be
  // the full extent of the strided area, thus sim->n*sim->NbofScalarfields
  std::cout << "creating fields with strided access\n";
#ifdef Viskores_ENABLE_CUDA
  std::cout << "copying single block data to device\n";
  T *device_AOS  = device_alloc(sim->n * sim->NbofScalarfields, &sim->scalarsAOS[0].mass);
  auto AOS = viskores::cont::make_ArrayHandle<T>(device_AOS,
                                             sim->n * sim->NbofScalarfields,
                                             viskores::CopyFlag::Off);
#else
  std::cout << "using host-allocated data\n";
// base of the AOS struct{}. every field will be offset from that base
  auto AOS = viskores::cont::make_ArrayHandle<T>(&sim->scalarsAOS[0].mass,
                                             sim->n * sim->NbofScalarfields,
                                             viskores::CopyFlag::Off);
#endif
  // starting at offset 0
  viskores::cont::ArrayHandleStride<T>  aos0(AOS, sim->n, sim->NbofScalarfields, 0);
  // coordinates are at offset 1, 2, 3 (x, y, z)
  viskores::cont::ArrayHandleStride<T> pos_x(AOS, sim->n, sim->NbofScalarfields, 1);
  viskores::cont::ArrayHandleStride<T> pos_y(AOS, sim->n, sim->NbofScalarfields, 2);
  viskores::cont::ArrayHandleStride<T> pos_z(AOS, sim->n, sim->NbofScalarfields, 3);

  viskores::cont::ArrayHandleStride<T>    vx(AOS, sim->n, sim->NbofScalarfields, 4);
  viskores::cont::ArrayHandleStride<T>    vy(AOS, sim->n, sim->NbofScalarfields, 5);
  viskores::cont::ArrayHandleStride<T>    vz(AOS, sim->n, sim->NbofScalarfields, 6);
  viskores::cont::ArrayHandleStride<T>  aos7(AOS, sim->n, sim->NbofScalarfields, 7);
  viskores::cont::ArrayHandleStride<T>  aos8(AOS, sim->n, sim->NbofScalarfields, 8);
  // first method to view a vector field from its base components
  auto velArray = viskores::cont::make_ArrayHandleCompositeVector(vx, vy, vz);
  //viskores::cont::printSummary_ArrayHandle(velArray, std::cout);
#else
  std::cout << "creating fields with independent (stride=1) access\n";
//https://vtk-m.readthedocs.io/en/stable/basic-array-handles.html#ex-arrayhandlefromvector
/*
  auto coordsArray1 =
    viskores::cont::make_ArrayHandleSOA<viskores::Vec3f>({sim->x, sim->y, sim->z});
    
  viskores::cont::printSummary_ArrayHandle(coordsArray1, std::cout);
  viskores::cont::ArrayHandle<viskores::Vec3f> positions1;
  viskores::cont::ArrayCopyDevice(coordsArray1, positions1);
  std::cout << "COORDARRAY1 HANDLE" << std::endl;
  viskores::cont::printSummary_ArrayHandle(coordsArray1, std::cout);
  std::cout << "--------------------------" << std::endl;
  */
  /****************************************/
  // https://viskores.readthedocs.io/en/latest/fancy-array-handles.html
  
#ifdef Viskores_ENABLE_CUDA
  std::cout << __FILE__ << ": " << __LINE__ << ": copying individual fields data to device\n";
  T *device_aos0  = device_alloc(sim->n, sim->mass.data());
  T *device_pos_x = device_alloc(sim->n, sim->x.data());
  T *device_pos_y = device_alloc(sim->n, sim->y.data());
  T *device_pos_z = device_alloc(sim->n, sim->z.data());
  T *device_vx    = device_alloc(sim->n, sim->vx.data());
  T *device_vy    = device_alloc(sim->n, sim->vy.data());
  T *device_vz    = device_alloc(sim->n, sim->vz.data());
  T *device_aos7  = device_alloc(sim->n, sim->rho.data());
  T *device_aos8  = device_alloc(sim->n, sim->temp.data());

  auto aos0  = viskores::cont::make_ArrayHandle(device_aos0,  sim->n, viskores::CopyFlag::Off);
  auto pos_x = viskores::cont::make_ArrayHandle(device_pos_x, sim->n, viskores::CopyFlag::Off);
  auto pos_y = viskores::cont::make_ArrayHandle(device_pos_y, sim->n, viskores::CopyFlag::Off);
  auto pos_z = viskores::cont::make_ArrayHandle(device_pos_z, sim->n, viskores::CopyFlag::Off);
  auto vx    = viskores::cont::make_ArrayHandle(device_vx,    sim->n, viskores::CopyFlag::Off);
  auto vy    = viskores::cont::make_ArrayHandle(device_vy,    sim->n, viskores::CopyFlag::Off);
  auto vz    = viskores::cont::make_ArrayHandle(device_vz,    sim->n, viskores::CopyFlag::Off);
  auto aos7  = viskores::cont::make_ArrayHandle(device_aos7,  sim->n, viskores::CopyFlag::Off);
  auto aos8  = viskores::cont::make_ArrayHandle(device_aos8,  sim->n, viskores::CopyFlag::Off);
#else
  auto aos0  = viskores::cont::make_ArrayHandle<T>(sim->mass, viskores::CopyFlag::Off);
  auto pos_x = viskores::cont::make_ArrayHandle<T>(sim->x,    viskores::CopyFlag::Off);
  auto pos_y = viskores::cont::make_ArrayHandle<T>(sim->y,    viskores::CopyFlag::Off);
  auto pos_z = viskores::cont::make_ArrayHandle<T>(sim->z,    viskores::CopyFlag::Off);
  auto vx    = viskores::cont::make_ArrayHandle<T>(sim->vx,   viskores::CopyFlag::Off);
  auto vy    = viskores::cont::make_ArrayHandle<T>(sim->vy,   viskores::CopyFlag::Off);
  auto vz    = viskores::cont::make_ArrayHandle<T>(sim->vz,   viskores::CopyFlag::Off);
  auto aos7  = viskores::cont::make_ArrayHandle<T>(sim->rho,  viskores::CopyFlag::Off);
  auto aos8  = viskores::cont::make_ArrayHandle<T>(sim->temp, viskores::CopyFlag::Off);
#endif
  /*
  // first method to view a vector field from its base components
  auto velArray = viskores::cont::make_ArrayHandleCompositeVector(vx, vy, vz);
  viskores::cont::printSummary_ArrayHandle(velArray, std::cout);
  dataSet.AddPointField("velocity", velArray);
  */
  // second method to view a vector field from its base components
  auto velArray = viskores::cont::make_ArrayHandleSOA(vx, vy, vz);
  //viskores::cont::printSummary_ArrayHandle(velArray, std::cout);

#endif
  
  auto coordsArray2 =
    viskores::cont::make_ArrayHandleCompositeVector(pos_x, pos_y, pos_z);
  //std::cout << "COORDARRAY2 HANDLE" << std::endl;
  //viskores::cont::printSummary_ArrayHandle(coordsArray2, std::cout);
  //std::cout << "--------------------------" << std::endl;
  
  viskores::cont::ArrayHandle<viskores::Vec3f> positions2;
  viskores::cont::ArrayCopy(coordsArray2, positions2);
  //std::cout << "POSITIONS2 HANDLE" << std::endl;
  //viskores::cont::printSummary_ArrayHandle(positions2, std::cout);
  //std::cout << "--------------------------" << std::endl;

  viskores::cont::ArrayHandle<viskores::Id> connectivity;
  viskores::cont::ArrayCopy(viskores::cont::make_ArrayHandleIndex(static_cast<viskores::Id>(sim->n)), connectivity);
  
  viskores::IdComponent numberOfPointsPerCell = 1;
  dataSet = dataSetBuilder.Create(positions2, // use template line 230
                                  viskores::CellShapeTagVertex(),
                                  numberOfPointsPerCell,
                                  connectivity, "coords");

  dataSet.AddPointField("mass",     aos0);
  dataSet.AddPointField("x",        pos_x);
  dataSet.AddPointField("y",        pos_y);
  dataSet.AddPointField("z",        pos_z);
  dataSet.AddPointField("vx",       vx);
  dataSet.AddPointField("vy",       vy);
  dataSet.AddPointField("vz",       vz);
  dataSet.AddPointField("rho",      aos7);
  dataSet.AddPointField("temp",     aos8);
  dataSet.AddPointField("velocity", velArray);
  //dataSet.PrintSummary(std::cout);
}

void Execute(int it, int frequency, int rank, const std::string &testname, const std::string &FileName)
{
  std::ostringstream fname;
  fname.str("");
  if(it % frequency == 0)
    {
    if (!testname.compare("histsampling"))
      {
      fname << FileName << "." << std::setfill('0') << std::setw(2)
          << rank << "." << std::setfill('0') << std::setw(4)
          << it << ".vtk";
      Execute_HistSampling(fname.str());
      }
    else if (!testname.compare("thresholding"))
      {
      fname << FileName << "." << std::setfill('0') << std::setw(2)
          << rank << "." << std::setfill('0') << std::setw(4)
          << it << ".vtk";
      Execute_ThresholdPoints(fname.str());
      }
    else if (!testname.compare("compositing"))
      {
      fname << FileName << "." << std::setfill('0') << std::setw(2)
          << rank << "." << std::setfill('0') << std::setw(4)
          << it << ".vtk";
      Execute_CompositeVectors(fname.str());
      }
    else if (!testname.compare("rendering"))
      {
      fname << FileName << "." << std::setfill('0') << std::setw(2)
          << rank << "." << std::setfill('0') << std::setw(4)
          << it << ".png";
      Execute_Rendering(fname.str());
      }
    else if (!testname.compare("dumping"))
      {
      fname << FileName << "." << std::setfill('0') << std::setw(2)
          << rank << "." << std::setfill('0') << std::setw(4)
          << it << ".vtk";
      Execute_Dumping(fname.str());
      }
    }
}

void Execute_CompositeVectors(const std::string &filename)
{
  using AssocType = viskores::cont::Field::Association;
  viskores::filter::field_transform::CompositeVectors compositor;
  compositor.SetActiveField(0, "vx");
  compositor.SetActiveField(1, "vy");
  compositor.SetActiveField(2, "vz");
  compositor.SetOutputFieldName("vxvyvz");
  compositor.SetFieldsToPass({ "rho" });
  auto compositorDataSet = compositor.Execute(dataSet);

// writing to disk (optional, for debugging only)
  if(filename.c_str())
    {
    viskores::io::VTKDataSetWriter writer(filename.c_str());
    writer.SetFileTypeToBinary();

    writer.WriteDataSet(compositorDataSet);
    }
}

void Execute_HistSampling(const std::string &filename)
{
  /********** HistSampling *****************/
  using AssocType = viskores::cont::Field::Association;
  viskores::filter::resampling::HistSampling histsample;
  histsample.SetNumberOfBins(128);
  histsample.SetSampleFraction(.1);
  histsample.SetActiveField("rho", AssocType::Points);
  auto histsampleDataSet = histsample.Execute(dataSet);

  viskores::filter::geometry_refinement::ConvertToPointCloud topc;
  topc.SetFieldsToPass("rho");
  auto topcDataSet = topc.Execute(histsampleDataSet);
  
  //viskores::filter::field_transform::LogValues lgv;
  //lgv.SetBaseValueTo10();
  //lgv.SetActiveField("rho", AssocType::Points);
  //lgv.SetOutputFieldName("log(rho)");
  
  //auto lgvDataSet = lgv.Execute(topcDataSet);
    
// writing to disk (optional, for debugging only)
  if(filename.c_str())
    {
    std::cout << "writing histogram sampling output " << filename << std::endl;
    viskores::io::VTKDataSetWriter writer(filename.c_str());
    writer.SetFileTypeToBinary();
    writer.WriteDataSet(topcDataSet);
    }
}

void Execute_Rendering(const std::string &filename)
{
  /*
  using AssocType = viskores::cont::Field::Association;
  viskores::filter::resampling::HistSampling histsample;
  histsample.SetNumberOfBins(128);
  histsample.SetSampleFraction(.95);
  histsample.SetActiveField("rho", AssocType::Points);
  auto histsampleDataSet = histsample.Execute(dataSet);

  viskores::filter::geometry_refinement::ConvertToPointCloud topc;
  topc.SetFieldsToPass("rho");
  auto topcDataSet = topc.Execute(histsampleDataSet);
  */
  auto topcDataSet = dataSet;
  //Creating Actor
  viskores::cont::ColorTable colorTable("viridis");
  viskores::rendering::Actor actor(topcDataSet.GetCellSet(),
                               topcDataSet.GetCoordinateSystem(),
                               topcDataSet.GetField("rho"),
                               colorTable);

  scene.AddActor(actor);

  mapper.SetUsePoints();
  // use radius = 0.05 for the small Tipsy example
  mapper.SetRadius(0.005f);
  mapper.UseVariableRadius(false);
  mapper.SetRadiusDelta(0.05f);

  viskores::rendering::View3D view(scene, mapper, canvas);

  bounds = topcDataSet.GetCoordinateSystem().GetBounds();
  if((bounds.X.Min < -1000.0) && (bounds.X.Max > 1000.0))
    {// use B=50 for the small Tipsy example
    float B=50.0;
    bounds = viskores::Bounds(viskores::Vec3f_64(-B, -B, -B),
                        viskores::Vec3f_64(B, B, B));
    }
  view.GetCamera().ResetToBounds(bounds);
  view.GetCamera().Azimuth(30.0);
  view.GetCamera().Elevation(30.0);
  view.SetBackgroundColor(viskores::rendering::Color(1.0f, 1.0f, 1.0f));
  view.SetForegroundColor(viskores::rendering::Color(0.0f, 0.0f, 0.0f));
  view.Paint();
  view.SaveAs(filename);
  std::cout << "written image to disk: "<< filename.c_str() << std::endl;
}

void Execute_ThresholdPoints(const std::string &filename)
{
  viskores::filter::entity_extraction::ThresholdPoints thresholdPoints;

  thresholdPoints.SetThresholdBetween(-0.1, 0.1);
  thresholdPoints.SetActiveField("z");
  thresholdPoints.SetFieldsToPass("rho");
  thresholdPoints.SetCompactPoints(true);
  auto output = thresholdPoints.Execute(dataSet);

// writing to disk (optional, for debugging only)
  if(filename.c_str())
    {
    std::cout << "writing threshold output to " << filename << std::endl;
    viskores::io::VTKDataSetWriter writer(filename.c_str());
    writer.SetFileTypeToBinary();
    writer.WriteDataSet(output);
    }
}

void Execute_Dumping(const std::string &filename)
{
  if(filename.c_str())
    {
    std::cout << "writing dumping output to " << filename << std::endl;
    viskores::io::VTKDataSetWriter writer(filename.c_str());
    writer.SetFileTypeToBinary();
    writer.WriteDataSet(dataSet);
    }
}

void Finalize(int &rank)
{
  std::cout << "Shutting down Viskores at end of processing\n";
}
}
#endif

