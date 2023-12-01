# Projective Dynamics with Dry Frictional Contact

![](https://hal.archives-ouvertes.fr/hal-02563307v2/file/projectiveDynamicsDryFrictionalContact.jpg)

This repository contains the code associated with the paper
[Projective Dynamics with Dry Frictional Contact](https://hal.archives-ouvertes.fr/hal-02563307)
published at SIGGRAPH 2020.

The code and data necessary to replicate the figures in the article is in the
replicability folder.

## Compilation

### Submodules

If you have not cloned the repository with the command
`git clone --recurse-submodules`, you have to update the submodules. To do this,
run the following commands:

```
git submodule init
git submodule update
```

### Install dependencies :

- Eigen (tested with 3.3.7)
- Doxygen _optional_ For building the documentation.
- Cgal
- Boost
- Openmp _optional_

### Compilation of the project

To compile the project, run the following commands:

```
mkdir build
cd build
cmake ..
make
```

Note that by default the build type is Debug. If you want to build in release
mode, instead of `cmake ..` run

```
cmake -DCMAKE_BUILD_TYPE=Release ..
```

## Usage

One argument only is expected:

- Path to a json configuration file. You can find documentation on the structure
  of the JSON configuration file [here](./Configuration.md). Some json
  configuration files can be found in
  [the `scenes` directory of this repository](./scenes).

```
./ProjectiveFriction path/to/configuration_file.json
```

### Quick start example

For example, to run the scenario which produces the teaser of the paper, enter
the following commands (it will take around 20-30 minutes to complete):

```
cd replicability
./ProjectiveFriction ./scenes/Square3/mu_0.1.json > /dev/null
```

### Output

The executable will create a directory `dataX` in the current directory, where
`X` is the smaller number greater or equal than 0 for which there doesn't
already exists a `dataX` directory. This means that each run of the `./fpd` will
create a new directory. This way, you are not overwriting your previous
simulation data.

After the executable has terminated, this directory has the following structure:

```
datax/
  configuration_file.json
  stats.txt
  output/
    out_XXXXXX.obj
```

- `configuration_file.json` is a copy of the configuration file passed to the
  executable.
- `stats.txt` contains data concerning the execution time of the simulation as
  well as the number of self-collision. See
  [the related documentation](./Stats.md) for more detail on its structure.
- `output/` contains `obj` file describing the state of the simulated mesh at
  each time frame.

### Looking at the simulation

If you want to see an animation of the simulation, we suggest the following
method (there are probably other suitable methods). For that you will need to
install

- [Paraview](https://www.paraview.org/), which is likely to be already present
  on the package repositories of your Linux distribution.
- [`obj2vtk`](https://gitlab.inria.fr/elan-public-code/obj2vtk), which you can
  install from its repository.

First convert the `obj` files produced by the executable to `vtk` files using
`obj2vtk`:

```
obj2vtk dataX/output/*.obj
```

Then run Paraview

```
paraview dataX/output/out_..vtp
```

and click the play button at the top of the Paraview GUI.

You can also add the meshes of the obstacle you used in your simulation. To do
this, just open the `obj` files in Paraview (`obj` files work in Paraview for
static objects, you cannot visualize a series of `obj` files within Paraview;
that is why we use `obj2vtk`).
