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
#include "physics/StaticAxisAlignedBox.hpp"

#include <boost/iterator/counting_iterator.hpp>

StaticAxisAlignedBox::StaticAxisAlignedBox(const Eigen::Vector3d& position,
                                           const Eigen::Vector3d& size,
                                           std::size_t material_identifier) :
  StaticAxisAlignedBox(Eigen::Vector3d(position),
                       Eigen::Vector3d(size),
                       material_identifier)
{
}

StaticAxisAlignedBox::StaticAxisAlignedBox(Eigen::Vector3d&& position,
                                           Eigen::Vector3d&& size,
                                           std::size_t material_identifier) :
  StaticObstacle(material_identifier),
  m_position(std::move(position)), m_size(std::move(size))
{
}

bool 
StaticAxisAlignedBox::checkCollision(const Eigen::Vector3d &pos,
	                                   Eigen::Vector3d &contact_point,
						                         Eigen::Vector3d &normal) const noexcept
{  
  Eigen::Vector3d relative = getRelative(pos);
  for (std::size_t i = 0; i < 3u; ++i)
  {
    if (std::abs(relative[i]) > m_size[i] / 2)
    {
      return false;
    }
  }
  
  contact_point =
    getAbsolute(getNearestPointOnSurfaceRelative(relative));
  normal = getSideNormal(getNearestSideRelative(relative));
  
  return true;
}


StaticAxisAlignedBox::Side
StaticAxisAlignedBox::getNearestSide(const Eigen::Vector3d& point) const
{
    return getNearestSideRelative(getRelative(point));
}

Eigen::Vector3d
StaticAxisAlignedBox::getNearestPointOnSurfaceRelative(
  const Eigen::Vector3d& point) const
{
    return getProjectionOnSideRelative(getNearestSideRelative(point), point);
}

StaticAxisAlignedBox::Side
StaticAxisAlignedBox::getNearestSideRelative(const Eigen::Vector3d& point) const
{
    return *std::min_element(
      getSideBegin(),
      getSideEnd(),
      std::bind(&StaticAxisAlignedBox::isSideNearerRelative,
                this,
                point,
                std::placeholders::_1,
                std::placeholders::_2));
}

bool
StaticAxisAlignedBox::isSideNearerRelative(const Eigen::Vector3d& point,
                                           Side lhs,
                                           Side rhs) const
{
    return getDistanceToSideRelative(lhs, point) <
           getDistanceToSideRelative(rhs, point);
}

Eigen::Vector3d
StaticAxisAlignedBox::getProjectionOnSideRelative(
  Side side,
  const Eigen::Vector3d& point) const
{
    assert(side != Side::NONE);
    Eigen::Vector3d result = point;
    result[getSideIndex(side)] =
      getSideSign(side) * 0.5 * m_size[getSideIndex(side)];
    return result;
}

double
StaticAxisAlignedBox::getDistanceToSideRelative(
  StaticAxisAlignedBox::Side side,
  const Eigen::Vector3d& point) const
{
    // We could have made this function simpler by using the distance to
    // the projection on the side, but this would have make it slower.
    assert(side != Side::NONE);
    return std::abs(point[getSideIndex(side)] -
                    getSideSign(side) * 0.5 * m_size[getSideIndex(side)]);
}

Eigen::Vector3d
StaticAxisAlignedBox::getRelative(const Eigen::Vector3d& point) const
{
    return point - m_position - 0.5 * m_size;
}

Eigen::Vector3d
StaticAxisAlignedBox::getAbsolute(const Eigen::Vector3d& point) const
{
    return point + m_position + 0.5 * m_size;
}

StaticAxisAlignedBox::SideIterator
StaticAxisAlignedBox::getSideBegin() const
{
    return StaticAxisAlignedBox::SideIterator(
      boost::counting_iterator<unsigned int>(
        static_cast<unsigned int>(Side::TOP)),
      staticCastFunction<Side, unsigned int>);
}

StaticAxisAlignedBox::SideIterator
StaticAxisAlignedBox::getSideEnd() const
{
    return StaticAxisAlignedBox::SideIterator(
      boost::counting_iterator<unsigned int>(
        static_cast<unsigned int>(Side::NONE)),
      staticCastFunction<Side, unsigned int>);
}

Eigen::Vector3d
StaticAxisAlignedBox::getSideNormal(Side side)
{
    Eigen::Vector3d result = Eigen::Vector3d::Zero();
    result[getSideIndex(side)] = getSideSign(side);
    return result;
}

constexpr std::size_t
StaticAxisAlignedBox::getSideIndex(Side side)
{
    assert(side != Side::NONE);
    switch (side)
    {
        case Side::TOP:
        case Side::BOTTOM:
            return 1;
        case Side::LEFT:
        case Side::RIGHT:
            return 0;
        case Side::FRONT:
        case Side::BACK:
            return 2;
        default:
            // This case should never happen.
          throw std::runtime_error("Unexpected box side request");
          return 0; // To make the compiler happy
    }
}

constexpr int
StaticAxisAlignedBox::getSideSign(Side side)
{
    assert(side != Side::NONE);
    switch (side)
    {
        case Side::TOP:
        case Side::RIGHT:
        case Side::FRONT:
            return 1;
        case Side::BOTTOM:
        case Side::LEFT:
        case Side::BACK:
            return -1;
        default:
            // This case should never happen.
          throw std::runtime_error("Unexpected box side request");
          return 0; // To make the compiler happy
    }
}
