# Scenes

This directory contains different configurations files for the simulation. To
understand these configuration files' structure, see [the related
documentation](../Configuration.md).

## `arabesque.json`

A character within a dress performing an arabesque. This scene was used in the
paper to show the accurate handling of complexe scenarios.

The meshes of the character and the dress are under a private Inria license.
Please contact laurence.boissieux@inria.fr and florence.descoubes@inria.fr to
obtain a license.

## `analytic.json`

A sheet falling parallel to an incline plane. Since the motion of such a scene
can be derive analytically, this scene was used for the validation of our method
in the paper.

## `sphere_1_layer.json`

A sheet falling onto a moving sphere which is itself placed on a plane. This
scene was used in the paper to show the effect of coulomb friction: the sheet
sticks to the sphere rather than sliding on it.

## `sphere_3_layer.json`

Three sheets falling onto a moving sphere which is itself placed on a plane. This
scene was used in the paper to show the correct handling of self-contact within
our method.

## `tree.json`

A ribbon falling onto a an inclined plane. This scene was used in the paper to
show that our method does have different behavior depending on the friction
coefficient.
