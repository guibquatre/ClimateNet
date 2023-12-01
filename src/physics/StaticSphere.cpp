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
#include "physics/StaticSphere.hpp"

StaticSphere::StaticSphere(double radius,
                           const Eigen::Vector3d& center,
                           std::size_t material_identifier)
  : StaticSphere(radius, Eigen::Vector3d(center), material_identifier)
{
}

StaticSphere::StaticSphere(double radius, Eigen::Vector3d&& center, std::size_t material_identifier)
  : StaticObstacle(material_identifier), m_sphere(radius, center)
{
}

#include <iostream>
bool
StaticSphere::checkCollision(const Eigen::Vector3d& pos,
                             Eigen::Vector3d& contact_point,
                             Eigen::Vector3d& normal) const noexcept
{
    if (!m_sphere.isInside(pos))
    {
        return false;
    }

    m_sphere.project(pos, contact_point, normal);
    return true;
}

