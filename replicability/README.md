# Replicability Stamp

This folder contains the files necessary for the [Graphics Replicability Stamp](https://www.replicabilitystamp.org/) procedure. It especially provides a single script to generate all the results (geometries and statistics) shown in the figures and tables of the paper, except the ones depending on a full character (non-free data). More precisely, with the script you will be able to generate the results presented in Fig. 1 (teaser), Fig. 3 and Table 2 (three first rows), Fig. 2, Fig. 5, and Fig. 6.

<i>Caveat</i>: As described in our [projective friction paper](https://inria.hal.science/hal-02563307v2), our method produces plausible stick-slip behaviours
driven by a friction coefficient. However, as the complexity of the scenarios
increases, the code may become indeterministic. This is probably due to the presence
of parallelism and the non-associativity of floating point operations. For
example, some scenarios shown in our paper are virtually impossible to
reproduce down to the exact detail with the script given in this folder. This
is the case of the first figure which shows three snapshots from a simulation of
cloth pieces dropped on a spinning sphere. Each run of the script associated to
this figure would give slightly different results, e.g., the cloth pieces might fall from the sphere at a different frame than the one shown in the paper. However, the
overall behaviour does remain consistent with the paper. Notably, a modification of the friction coefficient does alter the cloth behaviour as expected.

## Dependencies

The dependencies can be installed through
```
./dependencies.sh
```

## Compilation

The code can be compiled through
```
./compile.sh
```

## Generating the figures

The script `generate_figures.sh` generates all the figures of the paper except
for the ones with a character. The resulting files associated to a given figure
are stored in a directory `FigureX`, where `X` is the number of the figure.
See the following section for a description of each figure data.

### Figure 1

The folder contains the wavefront files that can be used to render Figure 1.
`left.obj` is the geometry of the render on the left of the figure. `right.obj`
the geometry of the render on the right. And, `middle.obj` is the geometry of
the render in the middle.

### Figure 2

The folder contains eight wavefront files. Each file has the form
`{left,right}_iterations_x.obj`. Files starting with `left` are associated with
the left render (the ribbons), files starting with `right` are associated with
the right render (the drape on a sphere). The value of `x` indicates the number
of local/global iterations used to obtain the obj.

Note that, `right_iterations_5.obj` can sometimes present no intersection with
the sphere, which is different from the figure in the paper. This is not an
issue, this just shows that, sometimes, the result is better than expected.

### Figure 3

The folder contains six wavefront files. Each file has the form
`row_<x>_{left,right}.obj`. Here, `<x>` is the number of the row of the render
associated to the file starting from the top. If the filename contains `left`,
then the associated render is on the left column of the figure, otherwise the
right column.

### Table 2

The folder contains the data for each row of Table 2. The data for a row is
stored in `<Scenario>/<Friction>/stats.txt`, where `<Scenario>` is the name of
the scenario in the paper and `<Friction>` is `mu<x>` where `<x>` is the
friction coefficient.

For every column we always consider the field `Mean Value`.  What follows is a
table which states under which section of the `stats.txt` file one can find the
value associated to a given column of Table 2. Note that
$`\bar{t}_{\mathrm{argus}}`$ and $`\bar{g}`$ do not have their own section as
producing them requires another software.

| Column Name                       | Section                              |
| -----------                       | -------                              |
| $`\bar{n}_\mathrm{ext}`$          | `Number of collisions`               |
| $`\bar{n}_\mathrm{self}`$         | `Number of self collisions`          |
| $`\bar{t}_\mathrm{rhs}`$          | `Full RHS computation time (µs)`     |
| $`\bar{t}_\mathrm{ext}`$          | `RHS friction overhead (µs)`         |
| $`\bar{t}_\mathrm{self}`$         | `Self collision time (µs)`           |
| $`\bar{t}_\mathrm{solve}`$        | `Solver time (µs)`                   |
| $`\bar{t}_i`$                     | `Iteration time (µs)`                |
| $`\bar{t}_\mathrm{contact}`$      | `Collision detection time (µs)`      |
| $`\bar{t}_\mathrm{self-contact}`$ | `Self collision detection time (µs)` |
| $`\bar{t}_\mathrm{sorting}`$      | `Self collision ordering time (µs)`  |
| $`\bar{t}_p`$                     | `Step time (µs)`                     |

### Figure 5

The folder contains one image file, `plots.png`, which shows the comparison
between simulation and analytical derivation of the velocity and the position.

### Figure 6

The folder contains one image file, `plots.png`, which shows the convergence
graph.
