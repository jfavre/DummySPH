# thresholding, i.e clipping the particle cloud half-way in the Z direction

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

reader = TrivialProducer(registrationName="grid")

# Create a new 'Render View'
renderView1 = GetRenderView()
renderView1.ViewSize = [1024, 1024]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [1.7854218337487884, 1.6179981242178436, 0.5]
renderView1.CameraPosition = [0.0, 0.0, 6.69]
renderView1.CameraFocalPoint = [0.0, 0.0, 0.0]
renderView1.CameraViewUp = [0.0, 1.0, 0.0]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 1.73205

outline1 = Outline(registrationName='Outline1', Input=reader)
outline1Display = Show(outline1)

clip1 = Clip(registrationName='Clip1', Input=reader)
clip1.ClipType = 'Plane'
clip1.ClipType.Normal = [0.0, 0.0, 1.0]

varname = "velocity"
readerDisplay = Show(clip1, renderView1, 'GeometryRepresentation')
readerDisplay.Representation = 'Point Gaussian'
ColorBy(readerDisplay, ['POINTS', varname])
readerDisplay.PointSize = 1.0
readerDisplay.GaussianRadius = 0.02

varnameLUT = GetColorTransferFunction(varname)
varnameLUT.ApplyPreset('Inferno (matplotlib)', True)
#varnameLUT.AutomaticRescaleRangeMode = "Grow and update every timestep"

varnameLUTColorBar = GetScalarBar(varnameLUT, renderView1)
varnameLUTColorBar.Title = varname
varnameLUTColorBar.ComponentTitle = ''
varnameLUTColorBar.Visibility = 1
readerDisplay.SetScalarBarVisibility(renderView1, True)
readerDisplay.RescaleTransferFunctionToDataRange(False, True)

ResetCamera()
GetActiveCamera().Azimuth(30)
GetActiveCamera().Elevation(30)
Render()

pNG1 = CreateExtractor('PNG', renderView1, registrationName='PNG1')
pNG1.Trigger = 'TimeStep'
pNG1.Trigger.Frequency = 1
pNG1.Writer.FileName = 'RenderView1_{timestep:06d}{camera}.png'
pNG1.Writer.ImageResolution = [1024, 1024]
pNG1.Writer.Format = 'PNG'

convertToPointCloud1 = ConvertToPointCloud(registrationName='ConvertToPointCloud1', Input=reader)
convertToPointCloud1.CellGenerationMode = 'Polyvertex cell'

vTP1 = CreateExtractor('VTPD', convertToPointCloud1, registrationName='VTPD1')
vTP1.Trigger = 'TimeStep'
vTP1.Trigger.Frequency = 1
vTP1.Writer.FileName = 'dataset_{timestep:06d}.vtpd'

# Catalyst options
from paraview import catalyst
options = catalyst.Options()
options.GlobalTrigger = 'TimeStep'
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = 'TimeStep'
