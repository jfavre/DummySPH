<div align="left">
 <img src="docs/_static/img/dummysph.jpg" width="150px">
</div>

# DummySPH

DummySPH is a mini-app to test in situ visualization libraries for Smoothed Particle Hydrodynamics (SPH)

# SPH

Smoothed Particle Hydrodynamics (SPH) codes are a family of simulation codes
based on a mesh-less set of particles. Based on a purely Lagrangian technique,
SPH particles are free to move with fluids or deforming solid structures. It has
been successfully used for several decades to simulate free-surface flows, solid
mechanics, multi-phase, fluid-structure interaction and astrophysics.

### SPH particle data memory layouts

Creating ```DummySPH``` was motivated to enable the testing of different in-situ
visualization scenarios for SPH simulations. We consider two SPH production codes 
with different internal data organisation:
* [PKDGRAV3](https://bitbucket.org/dpotter/pkdgrav3) uses an array-of-structures (AOS) data layout, while
* [SPH-EXA](https://github.com/unibas-dmi-hpc/SPH-EXA) uses a structure-of-arrays (SOA) data layout.

### in-situ visualization backends

Three different backend in-situ visualization libraries are enabled via compile-time flags.
* [Ascent](https://ascent.readthedocs.io/en/latest/index.html)
* [ParaView Catalyst](https://kitware.github.io/paraview-catalyst/)
* [VTK-m](https://vtk-m.readthedocs.io/en/stable/index.html)

#### Compilation

For proper compilation and installation of the in-situ backends, we refer the user
to the respective web sites of [Ascent](https://ascent.readthedocs.io/en/latest/index.html), [ParaView Catalyst](https://kitware.github.io/paraview-catalyst/), and [VTK-m](https://vtk-m.readthedocs.io/en/stable/index.html). For proper CUDA support, Ascent and VTK-m should be compiled accordingly.

Minimal CMake configuration:
```shell
mkdir build
cd build
cmake -S <GIT_SOURCE_DIR>/src
```

#### Running the main application

```DummySPH``` can start by generating a set of very simple data particle points arranged
in a cubic box of dimensions NxNxN, or by reading the output of the SPH applications which
motivated its creation, i.e. the SPH-EXA output in H5part format, or the PKDGRAV3 output in Tipsy format.
The last two options are enabled at compile time.

## Authors (in alphabetical order)

* Jean M. Favre
* Jean-Guillaume Piccinali

## Paper references

* Jean M. Favre, Jean-Guillaume Piccinali, Issues and challenges of deploying in-situ visualization for SPH codes, [WOIV'25, 9th International Workshop on In Situ Visualization](https://woiv.gitlab.io/woiv25/).
