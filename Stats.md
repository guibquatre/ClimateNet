# Stats

When the software is run a `stats.txt` file containing details on the execution
time of the simulation and the number of self-collision is produced (See [usage
documentation](./README.md). In this document, we present the structure of the
`stats.txt` file.

The file is made of sections separated by dashes
```
First section
------
Second section
------
...
```

We will describe each section in the following sections.

In every one of these section, the value associated to `Number` is the number of
time the process described in the section is done throughout the simulation.

## LHS computation time

This section gives the computation time in microseconds of the matrix of the
global system of Projective Dynamics.

## Cholesky computation time

This section gives the computation time in microseconds of the cholesky
factorization of the matrix of the global system of Projective Dynamics.

## Iteration time

This section gives an aggregate of the computation times in microseconds of the
Projective Dynamics local-global iterations. Note that his is different from the
computation of one time step as there are multiple local-global iterations per
time step.

## Collision detection time

This section gives an aggregate of the computation times in microseconds of the
colision detection. Note that this only contains collision detection, the
computation time of the resolution of these collisions is not contains here.
Also, note that collision detection is not part of our contribution, hence, the
time presented here does not show the performance of our contribution.

## Number of self collisions

When self-collision is enabled (See [the documentation on the
configuration](./Configuration.md)), this shows the aggregate of the number of
self-collisions detected during collision detection.

When self-collision is not enabled, `(No data)` is written.

## Solver time

This section gives an aggregate of the computation times in microseconds of the
solves of the global system of Projective Dynamics.

## Full RHS computation time

This section gives an aggregate of the computation times in microseconds of the
vector of the global system of Projective Dynamics. Hence, this is the local
step. This computation time contains both the computation time of the
contribution of the Projective Dynamics energies and the contribution of the
friction. 

## Step time

This section gives an aggregate of the computation times in microseconds of the
time-steps.

## RHS base computation time

This section gives an aggregate of the computation times in microseconds of the
contribution of the energies to the vector of the global system of Projective
Dynamics. 

## RHS friction overhead

This section gives an aggregate of the computation times in microseconds of the
contribution of the friction to the vector of the global system of Projective
Dynamics. This part of the simulation is our contribution.

Note that the value associated to `Number` might be smaller than the number of
local-global iterations in the simulation. This is because the friction is not
computed when there is no collision.
