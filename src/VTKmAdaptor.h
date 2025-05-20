#ifndef VTKmAdaptor_h
#define VTKmAdaptor_h

#include <iomanip>
#include <iostream>
#include <string>

#include <vtkm/cont/ColorTable.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/ArrayCopy.h>
#include <vtkm/cont/ArrayCopyDevice.h>
#include <vtkm/cont/DataSetBuilderExplicit.h>
#include <vtkm/cont/ArrayHandleExtractComponent.h>
#include <vtkm/cont/ArrayHandleCompositeVector.h>
#include <vtkm/cont/ArrayHandleStride.h>
#include <vtkm/cont/ArrayHandleIndex.h>
#include <vtkm/cont/Initialize.h>
#include <vtkm/io/VTKDataSetWriter.h>
#include <vtkm/rendering/Actor.h>
#include <vtkm/rendering/CanvasRayTracer.h>
#include <vtkm/filter/field_transform/LogValues.h>
#include <vtkm/filter/field_transform/CompositeVectors.h>
#include <vtkm/rendering/MapperPoint.h>
#include <vtkm/filter/resampling/HistSampling.h>
#include <vtkm/filter/geometry_refinement/ConvertToPointCloud.h>
#include <vtkm/filter/entity_extraction/ThresholdPoints.h>
#include <vtkm/rendering/Scene.h>
#include <vtkm/rendering/View3D.h>
/****************************************

three methods have been identified to create coordinates array handles from the
given std::vector<T> x, y, z;         // Positions

1) vtkm::cont::ArrayHandle<vtkm::Vec3f> coordsArray0;
   coordsArray0.Allocate(static_cast<vtkm::Id>(sim->n));

2) coordsArray1 = vtkm::cont::make_ArrayHandleSOA<vtkm::Vec3f>({sim->x, sim->y, sim->z});;

3) coordsArray2 = vtkm::cont::make_ArrayHandleCompositeVector(component1, component2, component3);

****************************************/

namespace VTKmAdaptor
{
  vtkm::rendering::CanvasRayTracer   canvas(1024,1024);
  vtkm::rendering::Scene             scene;
  vtkm::rendering::MapperPoint       mapper;
  vtkm::cont::DataSet                dataSet;
  vtkm::Bounds bounds;

void Execute_CompositeVectors(const std::string &filename);
void Execute_HistSampling(const std::string &filename);
void Execute_Rendering(const std::string &filename);
void Execute_ThresholdPoints(const std::string &filename);
void Execute_Dumping(const std::string &filename);

#ifdef VTKm_ENABLE_CUDA
template<typename T>
T *
device_alloc(int size, const T *data)
{
  void *buff;
  cudaMalloc(&buff, size * sizeof(T));
  cudaMemcpy(buff, data, size * sizeof(T), cudaMemcpyHostToDevice);
  return static_cast<T*>(buff);
}
#endif

template<typename T>
void Initialize(int argc, char* argv[], sph::ParticlesData<T> *sim)
{
  std::cout << "VTK-m::Initialize" << std::endl;

#ifdef VTKm_ENABLE_CUDA
  std::cout << "forcing DeviceAdapterTagCuda" << std::endl;
  vtkm::cont::ScopedRuntimeDeviceTracker(vtkm::cont::DeviceAdapterTagCuda{});
#endif

  vtkm::cont::Initialize(argc, argv);

  vtkm::cont::DataSetBuilderExplicit dataSetBuilder;
  /*
  vtkm::cont::ArrayHandle<vtkm::Vec3f> coordsArray0;
  coordsArray0.Allocate(static_cast<vtkm::Id>(sim->n));

  auto coordsPortal = coordsArray0.WritePortal();
  for (std::size_t index = 0; index < sim->n; ++index)
  {
    coordsPortal.Set(static_cast<vtkm::Id>(index),
                     vtkm::make_Vec(static_cast<vtkm::FloatDefault>(sim->x[index]),
                                    static_cast<vtkm::FloatDefault>(sim->y[index]),
                                    static_cast<vtkm::FloatDefault>(sim->z[index])));
  }
  std::cout << "COORDARRAY0 HANDLE" << std::endl;
  vtkm::cont::printSummary_ArrayHandle(coordsArray0, std::cout);
  std::cout << "--------------------------" << std::endl;
  */
  /****************************************/
#ifdef STRIDED_SCALARS
  // NOTE: In this case, the num_vals, needs to be
  // the full extent of the strided area, thus sim->n*sim->NbofScalarfields
  std::cout << "creating fields with strided access\n";
#ifdef VTKm_ENABLE_CUDA
  std::cout << "copying single block data to device\n";
  T *device_AOS  = device_alloc(sim->n * sim->NbofScalarfields, &sim->scalarsAOS[0].mass);
  auto AOS = vtkm::cont::make_ArrayHandle<T>(device_AOS,
                                             sim->n * sim->NbofScalarfields,
                                             vtkm::CopyFlag::Off);
  sim->scalarsAOS.clear(); // just to make sure we're going to the device
#ifdef SPH_DOUBLE
  sim->scalarsAOS = std::vector<sph::ParticlesData<double>::tipsySph>();
#else
  sim->scalarsAOS = std::vector<sph::ParticlesData<float>::tipsySph>();
#endif
#else
  std::cout << "using host-allocated data\n";
// base of the AOS struct{}. every field will be offset from that base
  auto AOS = vtkm::cont::make_ArrayHandle<T>(&sim->scalarsAOS[0].mass,
                                             sim->n * sim->NbofScalarfields,
                                             vtkm::CopyFlag::Off);
#endif
  // starting at offset 0
  vtkm::cont::ArrayHandleStride<T>  aos0(AOS, sim->n, sim->NbofScalarfields, 0);
  // coordinates are at offset 1, 2, 3 (x, y, z)
  vtkm::cont::ArrayHandleStride<T> pos_x(AOS, sim->n, sim->NbofScalarfields, 1);
  vtkm::cont::ArrayHandleStride<T> pos_y(AOS, sim->n, sim->NbofScalarfields, 2);
  vtkm::cont::ArrayHandleStride<T> pos_z(AOS, sim->n, sim->NbofScalarfields, 3);

  vtkm::cont::ArrayHandleStride<T>    vx(AOS, sim->n, sim->NbofScalarfields, 4);
  vtkm::cont::ArrayHandleStride<T>    vy(AOS, sim->n, sim->NbofScalarfields, 5);
  vtkm::cont::ArrayHandleStride<T>    vz(AOS, sim->n, sim->NbofScalarfields, 6);
  vtkm::cont::ArrayHandleStride<T>  aos7(AOS, sim->n, sim->NbofScalarfields, 7);
  vtkm::cont::ArrayHandleStride<T>  aos8(AOS, sim->n, sim->NbofScalarfields, 8);
  // first method to view a vector field from its base components
  auto velArray = vtkm::cont::make_ArrayHandleCompositeVector(vx, vy, vz);
  //vtkm::cont::printSummary_ArrayHandle(velArray, std::cout);
#else
  std::cout << "creating fields with independent (stride=1) access\n";
//https://vtk-m.readthedocs.io/en/stable/basic-array-handles.html#ex-arrayhandlefromvector
/*
  auto coordsArray1 =
    vtkm::cont::make_ArrayHandleSOA<vtkm::Vec3f>({sim->x, sim->y, sim->z});
    
  vtkm::cont::printSummary_ArrayHandle(coordsArray1, std::cout);
  vtkm::cont::ArrayHandle<vtkm::Vec3f> positions1;
  vtkm::cont::ArrayCopyDevice(coordsArray1, positions1);
  std::cout << "COORDARRAY1 HANDLE" << std::endl;
  vtkm::cont::printSummary_ArrayHandle(coordsArray1, std::cout);
  std::cout << "--------------------------" << std::endl;
  */
  /****************************************/
  // https://vtk-m.readthedocs.io/en/v2.2.0/fancy-array-handles.html#composite-vector-arrays
  
#ifdef VTKm_ENABLE_CUDA
  std::cout << "copying individual fields data to device\n";
  T *device_aos0  = device_alloc(sim->n, sim->mass.data());
  T *device_pos_x = device_alloc(sim->n, sim->x.data());
  T *device_pos_y = device_alloc(sim->n, sim->y.data());
  T *device_pos_z = device_alloc(sim->n, sim->z.data());
  T *device_vx    = device_alloc(sim->n, sim->vx.data());
  T *device_vy    = device_alloc(sim->n, sim->vy.data());
  T *device_vz    = device_alloc(sim->n, sim->vz.data());
  T *device_aos7  = device_alloc(sim->n, sim->rho.data());
  T *device_aos8  = device_alloc(sim->n, sim->temp.data());
  
  auto aos0  = vtkm::cont::make_ArrayHandle(device_aos0,  sim->n, vtkm::CopyFlag::Off);
  auto pos_x = vtkm::cont::make_ArrayHandle(device_pos_x, sim->n, vtkm::CopyFlag::Off);
  auto pos_y = vtkm::cont::make_ArrayHandle(device_pos_y, sim->n, vtkm::CopyFlag::Off);
  auto pos_z = vtkm::cont::make_ArrayHandle(device_pos_z, sim->n, vtkm::CopyFlag::Off);
  auto vx    = vtkm::cont::make_ArrayHandle(device_vx,    sim->n, vtkm::CopyFlag::Off);
  auto vy    = vtkm::cont::make_ArrayHandle(device_vy,    sim->n, vtkm::CopyFlag::Off);
  auto vz    = vtkm::cont::make_ArrayHandle(device_vz,    sim->n, vtkm::CopyFlag::Off);
  auto aos7  = vtkm::cont::make_ArrayHandle(device_aos7,  sim->n, vtkm::CopyFlag::Off);
  auto aos8  = vtkm::cont::make_ArrayHandle(device_aos8,  sim->n, vtkm::CopyFlag::Off);
#else
  auto aos0  = vtkm::cont::make_ArrayHandle<T>(sim->mass, vtkm::CopyFlag::Off);
  auto pos_x = vtkm::cont::make_ArrayHandle<T>(sim->x,    vtkm::CopyFlag::Off);
  auto pos_y = vtkm::cont::make_ArrayHandle<T>(sim->y,    vtkm::CopyFlag::Off);
  auto pos_z = vtkm::cont::make_ArrayHandle<T>(sim->z,    vtkm::CopyFlag::Off);
  auto vx    = vtkm::cont::make_ArrayHandle<T>(sim->vx,   vtkm::CopyFlag::Off);
  auto vy    = vtkm::cont::make_ArrayHandle<T>(sim->vy,   vtkm::CopyFlag::Off);
  auto vz    = vtkm::cont::make_ArrayHandle<T>(sim->vz,   vtkm::CopyFlag::Off);
  auto aos7  = vtkm::cont::make_ArrayHandle<T>(sim->rho,  vtkm::CopyFlag::Off);
  auto aos8  = vtkm::cont::make_ArrayHandle<T>(sim->temp, vtkm::CopyFlag::Off);
#endif
  /*
  // first method to view a vector field from its base components
  auto velArray = vtkm::cont::make_ArrayHandleCompositeVector(vx, vy, vz);
  vtkm::cont::printSummary_ArrayHandle(velArray, std::cout);
  dataSet.AddPointField("velocity", velArray);
  */
  // second method to view a vector field from its base components
  auto velArray = vtkm::cont::make_ArrayHandleSOA(vx, vy, vz);
  //vtkm::cont::printSummary_ArrayHandle(velArray, std::cout);

#endif
  
  auto coordsArray2 =
    vtkm::cont::make_ArrayHandleCompositeVector(pos_x, pos_y, pos_z);
  //std::cout << "COORDARRAY2 HANDLE" << std::endl;
  //vtkm::cont::printSummary_ArrayHandle(coordsArray2, std::cout);
  //std::cout << "--------------------------" << std::endl;
  
  vtkm::cont::ArrayHandle<vtkm::Vec3f> positions2;
  vtkm::cont::ArrayCopy(coordsArray2, positions2);
  //std::cout << "POSITIONS2 HANDLE" << std::endl;
  //vtkm::cont::printSummary_ArrayHandle(positions2, std::cout);
  //std::cout << "--------------------------" << std::endl;

  vtkm::cont::ArrayHandle<vtkm::Id> connectivity;
  vtkm::cont::ArrayCopy(vtkm::cont::make_ArrayHandleIndex(static_cast<vtkm::Id>(sim->n)), connectivity);
  
  vtkm::IdComponent numberOfPointsPerCell = 1;
  dataSet = dataSetBuilder.Create(positions2, // use template line 230
                                  vtkm::CellShapeTagVertex(),
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
  using AssocType = vtkm::cont::Field::Association;
  vtkm::filter::field_transform::CompositeVectors compositor;
  compositor.SetActiveField(0, "vx");
  compositor.SetActiveField(1, "vy");
  compositor.SetActiveField(2, "vz");
  compositor.SetOutputFieldName("vxvyvz");
  compositor.SetFieldsToPass({ "vxvyvz", "velocity", "rho" });
  auto compositorDataSet = compositor.Execute(dataSet);

// writing to disk (optional, for debugging only)
  if(filename.c_str())
    {
    vtkm::io::VTKDataSetWriter histsampleWriter(filename.c_str());
    histsampleWriter.SetFileTypeToBinary();

    histsampleWriter.WriteDataSet(compositorDataSet);
    }
}

void Execute_HistSampling(const std::string &filename)
{
  /********** HistSampling *****************/
  using AssocType = vtkm::cont::Field::Association;
  vtkm::filter::resampling::HistSampling histsample;
  histsample.SetNumberOfBins(128);
  histsample.SetSampleFraction(.01);
  histsample.SetActiveField("rho", AssocType::Points);
  auto histsampleDataSet = histsample.Execute(dataSet);

  vtkm::filter::geometry_refinement::ConvertToPointCloud topc;
  topc.SetFieldsToPass("rho");
  auto topcDataSet = topc.Execute(histsampleDataSet);
  
  //vtkm::filter::field_transform::LogValues lgv;
  //lgv.SetBaseValueTo10();
  //lgv.SetActiveField("rho", AssocType::Points);
  //lgv.SetOutputFieldName("log(rho)");
  
  //auto lgvDataSet = lgv.Execute(topcDataSet);
    
// writing to disk (optional, for debugging only)
  if(filename.c_str())
    {
    std::cout << "writing histogram sampling output " << filename << std::endl;
    vtkm::io::VTKDataSetWriter histsampleWriter(filename.c_str());
    histsampleWriter.SetFileTypeToBinary();
    histsampleWriter.WriteDataSet(topcDataSet);
    }
}

void Execute_Rendering(const std::string &filename)
{
  using AssocType = vtkm::cont::Field::Association;
  vtkm::filter::resampling::HistSampling histsample;
  histsample.SetNumberOfBins(128);
  histsample.SetSampleFraction(.2);
  histsample.SetActiveField("rho", AssocType::Points);
  auto histsampleDataSet = histsample.Execute(dataSet);

  vtkm::filter::geometry_refinement::ConvertToPointCloud topc;
  topc.SetFieldsToPass("rho");
  auto topcDataSet = topc.Execute(histsampleDataSet);
  
  //Creating Actor
  vtkm::cont::ColorTable colorTable("viridis");
  vtkm::rendering::Actor actor(topcDataSet.GetCellSet(),
                               topcDataSet.GetCoordinateSystem(),
                               topcDataSet.GetField("rho"),
                               colorTable);

  scene.AddActor(actor);

  mapper.SetUsePoints();
  // use radius = 0.05 for the small Tipsy example
  mapper.SetRadius(0.005f);
  mapper.UseVariableRadius(false);
  mapper.SetRadiusDelta(0.05f);

  vtkm::rendering::View3D view(scene, mapper, canvas);

  bounds = topcDataSet.GetCoordinateSystem().GetBounds();
  if((bounds.X.Min < -1000.0) && (bounds.X.Max > 1000.0))
    {// use B=50 for the small Tipsy example
    float B=50.0;
    bounds = vtkm::Bounds(vtkm::Vec3f_64(-B, -B, -B),
                        vtkm::Vec3f_64(B, B, B));
    }
  view.GetCamera().ResetToBounds(bounds);
  view.GetCamera().Azimuth(30.0);
  view.GetCamera().Elevation(30.0);
  view.SetBackgroundColor(vtkm::rendering::Color(1.0f, 1.0f, 1.0f));
  view.SetForegroundColor(vtkm::rendering::Color(0.0f, 0.0f, 0.0f));
  view.Paint();
  view.SaveAs(filename);
  std::cout << "written image to disk: "<< filename.c_str() << std::endl;
}

void Execute_ThresholdPoints(const std::string &filename)
{
  vtkm::filter::entity_extraction::ThresholdPoints thresholdPoints;

  thresholdPoints.SetThresholdBetween(-0.1, 0.1);
  thresholdPoints.SetActiveField("z");
  thresholdPoints.SetFieldsToPass("rho");
  thresholdPoints.SetCompactPoints(true);
  auto output = thresholdPoints.Execute(dataSet);

// writing to disk (optional, for debugging only)
  if(filename.c_str())
    {
    std::cout << "writing threshold output to " << filename << std::endl;
    vtkm::io::VTKDataSetWriter writer(filename.c_str());
    writer.SetFileTypeToBinary();
    writer.WriteDataSet(output);
    }
}

void Execute_Dumping(const std::string &filename)
{
  if(filename.c_str())
    {
    std::cout << "writing dumping output to " << filename << std::endl;
    vtkm::io::VTKDataSetWriter writer(filename.c_str());
    writer.SetFileTypeToBinary();
    writer.WriteDataSet(dataSet);
    }
}

void Finalize(int &rank)
{
  std::cout << "Shutting down VTK-m at end of processing\n";
}
}
#endif

