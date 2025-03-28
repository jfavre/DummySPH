# DummySPH
a mini-app to test in situ visualization libraries for Smoothed Particle Hydrodynamics

# What's new

For VTK-m, cmake -DINSITU=VTK-m -DSTRIDED_SCALARS=ON <<other build options>>
we can now test different  techniques, using the Tipsy data output

./bin/dummysph_vtkm --tipsy hr8799_bol_bd1.017300 --thresholding /dev/shm/threshold
./bin/dummysph_vtkm --tipsy hr8799_bol_bd1.017300 --histsampling /dev/shm/histsampling
./bin/dummysph_vtkm --tipsy hr8799_bol_bd1.017300 --compositing /dev/shm/composite
./bin/dummysph_vtkm --tipsy hr8799_bol_bd1.017300 --rendering /dev/shm/image

For Ascent, cmake -DINSITU=Ascent  <<other build options>>

./bin/dummysph_ascent --thresholding /dev/shm/box_particles
./bin/dummysph_ascent --histsampling /dev/shm/particles_0.05
./bin/dummysph_ascent --compositing /dev/shm/composite
./bin/dummysph_ascent --rendering /dev/shm/image

a special case where the second parameter is the name of the variable for which we extract the min()and max() values
./bin/dummysph_ascent  --binning rho
