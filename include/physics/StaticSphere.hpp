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
#ifndef STATIC_SPHERE_HPP
#define STATIC_SPHERE_HPP

#include "physics/StaticObstacle.hpp"
#include "geometry/Sphere.hpp"

class StaticSphere : public StaticObstacle
{
public:
  StaticSphere(double radius,
               const Eigen::Vector3d& center,
               std::size_t material_identifier);
  StaticSphere(double radius,
               Eigen::Vector3d&& center,
               std::size_t material_indentifier);

  /**
   * @brief Check a collision
   * @param[in]  pos            Position to check
   * @param[out] contact_point  Contact point (if valid)
   * @param[out] normal         Normal at the contact point (if valid)
   * @return true if there is a contact (thus contact_point and normal are valid)
   */
  bool checkCollision(const Eigen::Vector3d &pos,
                      Eigen::Vector3d &contact_point,
                      Eigen::Vector3d &normal) const noexcept override final;
private:
  Sphere m_sphere;
};

#endif
