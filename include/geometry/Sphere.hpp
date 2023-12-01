/*
Copyright 2021 by Inria, Mickaël Ly, Jean Jouve, Florence Bertails-Descoubes and
    Laurence Boissieux

This file is part of ProjectiveFriction.

ProjectiveFriction is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

ProjectiveFriction is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
details.

You should have received a copy of the GNU General Public License along with
ProjectiveFriction. If not, see <https://www.gnu.org/licenses/>.
*/
#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "geometry/Mesh.hpp"
#include "types.hpp"

#include <Eigen/Core>

class Sphere
{
  public:
    Sphere(const Sphere& other) = default;
    Sphere(double radius, const Eigen::Vector3d& center);
    Sphere(double radius, Eigen::Vector3d&& center);

    Mesh getMesh(double number_sector, double number_stack) const noexcept;

    const Eigen::Vector3d& getCenter() const noexcept;
    double getRadius() const noexcept;

    bool isInside(const Eigen::Vector3d& point) const noexcept;
    /**
     *  @brief Stores the projection of \a point on the sphere surface in \a projection and the
     *  normal at the the projection in \a normal.
     */
    void project(const Eigen::Vector3d& point,
                 Eigen::Vector3d& projection,
                 Eigen::Vector3d& normal) const noexcept;

  private:
    double m_radius;
    Eigen::Vector3d m_center;
};

#endif
