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
#include "physics/StaticPlane.hpp"


StaticPlane::StaticPlane(const Eigen::Vector3d& normal,
                         const Eigen::Vector3d& point,
                         std::size_t material_identifier) :
  StaticPlane(Eigen::Vector3d(normal),
              Eigen::Vector3d(point),
              material_identifier)
{
}

StaticPlane::StaticPlane(const Eigen::Vector3d& normal,
                         Eigen::Vector3d&& point,
                         std::size_t material_identifier) :
  StaticPlane(Eigen::Vector3d(normal), std::move(point), material_identifier)
{
}

StaticPlane::StaticPlane(Eigen::Vector3d&& normal,
                         const Eigen::Vector3d& point,
                         std::size_t material_identifier) :
  StaticPlane(std::move(normal), Eigen::Vector3d(point), material_identifier)
{
}

StaticPlane::StaticPlane(Eigen::Vector3d&& normal,
                         Eigen::Vector3d&& point,
                         std::size_t material_identifier) :
  StaticObstacle(material_identifier),
  m_normal(std::move(normal)), m_point(std::move(point))
{
  m_normal.normalize();
}



bool 
StaticPlane::checkCollision(const Eigen::Vector3d &pos,
	                          Eigen::Vector3d &contact_point,
						                Eigen::Vector3d &normal) const noexcept
{
  const double distance_to_plane = (pos - m_point).dot(m_normal);
  const bool is_inside = (distance_to_plane < 1.5e-2);
  
  if (!is_inside) 
  {
    return false;
  }
  
  contact_point = pos - m_normal * distance_to_plane;
  normal = m_normal;
  return true;
}

