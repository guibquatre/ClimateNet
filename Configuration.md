# Configuration

The configuration of a scene and its simulation is done through a JSON file. The
structure of this JSON file is as follows. The root of the JSON file is an
object which must contain 3 keys:
- `"Parameters"` The parameters of the simulation.
- `"Meshes"` The objects whose dynamics will be computed.
- `"Obstacles"` The objects whose dynamics will **not** be computed. 
- `"Friction coefficients"` A matrix of the friction coefficients. 

```json
{
    "Parameters" : ...,
    "Meshes" : ...,
    "Obstacles" : ...,
    "Friction coefficients" : ...
}
```

## `"Parameters"`

The key `"Meshes"` must be associated to an object that contains the following
key
- `"Frame number"` Number of frame to compute. The simulation will stop after
  it has computed the indicated number of frame. Note that a frame is different
  from a time-step. Usually multiple time-steps are done to produce one frame.
- `"LocalGlobal"` Number of local-global Projective Dynamics iteration to do
  each time-step.
- `"TimeStep"` Step size in the simulation.
- `"Steps per frame"` Number of time-step to compute for each frame.
- `"Self-collision"` A boolean indicating if self-collision should be taken into
  account.
- `"Self-collision tolerance"` Only required if `"Self-collision"` is set to
  true. The collision detection tolerance for self-collision. The surfaces
  of two dynamic objects are considered in contact if they are at a distance
  less than the tolerance.
- `"Air Damping"` (Optional) Permanent viscous friction coefficient. This
  introduces a force $`-\alpha \mathbf{v}`$. This parameter defaults to `0.0`.
- `"Gravity"` An array of floating point number representing the gravitational
  field.

```json
"Parameters" : {
    "Frame number" : 50,
    "LocalGlobal" : 20,
    "TimeStep" : 1.0e-3,
    "Steps per frame" : 16,
    "Self-collision" : true,
    "Self-collision tolerance" : 1.0e-2,
    "Air Damping" : 0.1,
    "Gravity" : [ 0.0, 0.0, -9.81 ],
}
```

## `"Meshes"`

The key `"Meshes"` must be associated to an array of object. Each object defines
one dynamic entity in the simulation. **However, our simulation does not support**
**multiple dynamic mesh at the moment ie the array must be of size 1**. Each of 
these objects must have the following keys:
- `"Obj filename"` The path to an OBJ file that contains the rest state of the
  object.
- `"Area density"` The surfacic mass of the object.
- `"Bending"` The bending stiffness of the object.
- `"Stretching"` The stretching stiffness of the object.
- `"Material Identifier"` An identifier that defines the material of the object.
  This will be used to query the friction coefficients the object has with other
  materials.
These objects can optionaly have the key
- `"Current state obj filename"` The geometry of the object at the beginning of
  the simulation. The initial position of the vertices of the object will be set
  to the position of the vertices find in this file.

```json
"Meshes" : [
    {
        "Obj filename" : "path/to/file.obj",
        "Area density" : 1.0,
        "Bending" : 1.0,
        "Stretching" : 1.0,
        "Material Identifier" : 0,
        "Current state obj filename" : "path/to/other_file.obj"
    },
    ...
]
```

## `"Obstacles"`

The key `"Obstacles"` must be associated to an array of object. Each of these
object must contain the two keys
- `"Type"` The type of obstacle.
- `"Material Idenfier"` An identifier that defines the material of the object.
  This will be used to query the friction coefficients the object has with other
  materials.
The other required key depends on the value associated to `"Type"`. The key
`"Type"` can take the following value:
- `"Sphere"` Defines a static sphere.
- `"Plane"` Defines a static plane.
- `"Axis Aligned Box"` Defines a static axis aligned box.
- `"Mesh"` Defines a static obstacle from an obj file.
- `"Moving Mesh"` Defines a moving obstacle from a serie of obj file. The files
   represent the positions at key time, the movement in between is linearly
   interpolated.
- `"Rotating Sphere"` Define a sphere rotating around an axis.
The following sections define the other required keys for each type.

```json
"Obstacle" : [
    {
        "Type" : "The Type",
        "Material Identifier" : 0,
        ...
    },
    ...
]
```


### `"Sphere"`

An obstacle of type `"Sphere"` has the following required keys:
- `"Position"` An array of floating point number representing the center of the
  sphere.
- `"Radius"` The radius of the sphere.

```json
{
    "Type" : "Sphere",
    "Material Idenfier" : 0,
    "Position" : [ 0.0, 0.0, 0.0 ],
    "Radius" : 1.0
}
```

### `"Plane"`

An obstacle of type `"Plane"` has the following required keys:
- `"Point"` An array of floating point number representing a point by which the
  plane goes through.
- `"Normal"` An array of floating point number representing a vector orthogonal
  to the place.

```json
{
    "Type" : "Plane",
    "Material Idenfier" : 0,
    "Point" : [ 0.0, 0.0, 0.0 ],
    "Normal" : [ 1.0, 0.0, 0.0 ]
}
```

### `"Axis Aligned Box"`

An obstacle of type `"Axis Aligned Box"` has the following required keys:
- `"Position"` An array of floating point number representing the center of the
  box.
- `"Size"` An array of floating point number. The first number is the size of
  the side along the x axis, the second of the y axis and the third of the z
  axis.

```json
{
    "Type" : "Axis Aligned Box",
    "Material Idenfier" : 0,
    "Position" : [ 0.0, 0.0, 0.0 ],
    "Size" : [ 1.0, 1.0, 1.0 ]
}
```

### `"Mesh"`

An obstacle of type `"Mesh"` has the following required keys:
- `"Path"` The path to the OBJ file that stores the geometry of the obstacle.
- `"Threshold"` The threshold used to detext collision. If a point is at a
  distance less that this threshold from the surface of the mesh, its is
  considered inside.

```json
{
    "Type" : "Mesh",
    "Material Idenfier" : 0,
    "Path" : "path/to/file.obj",
    "Threshold" : 0.005
}
```

### `"Moving Mesh"`

An obstacle of type `"Moving Mesh"` has the following required keys:
- `"Obj Prefix"` The prefix of the OBJ file containing the mesh serie. If
  the file has the form `file_000456.obj`, the value should be `file_`.
- `"Suffix Size"` The number of digit used to represent the index of the file
  within the serie. If the file has the form `file_000456.obj`, the value should
  be `6`.
- `"Number Frame"` The number of file in the serie.
- `"Time Between Frame"` The time between each frame.
- `"Collision tolerance"` The tolerance of the collision detection. A point will
  be considered in collision with the obstacle if it is at a distance less than
  the associated value from the surface.
Between the frames, the mesh is interpolated linearly.

```json
{
    "Type" : "Moving Mesh",
    "Material Idenfier" : 0,
    "Obj Prefix" : "path/to/file_",
    "Suffix Size" : 6,
    "Number Frame" : 500,
    "Time Between Frame" : 0.01,
    "Collision tolerance" : 0.005
}
```

### `"Rotating Sphere"`

An obstacle of type `"Rotating Sphere"` has the following required keys:
- `"Position"` An array of floating point number representing the center of the
  sphere.
- `"Radius"` The radius of the sphere.
- `"Rotation Axe"` An array of floating point number representing axis around
  which the sphere rotates.
- `"Radial Speed"` The rotation speed in radiant per second.
- `"Rotation Start Time"` The simulation time at which the sphere will
  start rotating.

```json
{
    "Type" : "Rotating Sphere",
    "Material Idenfier" : 0,
    "Position" : [ 0.0, 0.0, 0.0 ],
    "Radius" : 1.0,
    "Rotation Axe" : [ 1.0, 0.0, 0.0 ],
    "Radial Speed" : 3.0,
    "Rotation Start Time" : 5.0
}
```

## `"Friction Coefficients"`

The value associated to the key `"Friction Coefficients"` must be an array of
array of floating point number. The value at position `[i][j]` in the array is
the friction coefficients between object with `"Material Identifier" : i` and
object with `"Material Identifier" : j`. Note that for this array of array to
make sense physically it must be symmetric, i.e the value at `[i][j]` is that
same as the value at `[j][i]`.

```json
"Friction Coefficients" : [
    [ 0.3, 0.1 ],
    [ 0.1, 0.2 ]
]
```
