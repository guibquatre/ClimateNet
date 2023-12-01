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
#ifndef STATIC_AXIS_ALIGNED_BOX_HPP
#define STATIC_AXIS_ALIGNED_BOX_HPP

#include <Eigen/Core>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>

#include "cpp_utils.hpp"
#include "physics/StaticObstacle.hpp"

class StaticAxisAlignedBox : public StaticObstacle
{
public:
    /**
     * The differents sides of the box. We use the same coordinate system
     * as OpenGL.
     */
    enum class Side
    {
        TOP,
        BOTTOM,
        LEFT,
        RIGHT,
        FRONT,
        BACK,
        NONE
    };

    using SideIterator =
      boost::transform_iterator<std::function<Side(unsigned int)>,
                                boost::counting_iterator<unsigned int>>;

    StaticAxisAlignedBox(const Eigen::Vector3d& position,
                         const Eigen::Vector3d& size,
                         std::size_t material_identifier);
    StaticAxisAlignedBox(Eigen::Vector3d&& position,
                         Eigen::Vector3d&& size,
                         std::size_t material_identifier);

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
  static Eigen::Vector3d getSideNormal(Side side);
  /**
   * Return the index of the corresponding axis.
   */
  static constexpr std::size_t getSideIndex(Side side);
  /**
   * Return 1 if the side is toward the increasing value of the corresponding
   * axis, -1 otherwise.
   */
  static constexpr int getSideSign(Side side);

  Side getNearestSide(const Eigen::Vector3d& point) const;
  /**
   * Return the nearest point on the surface as if the center of the
   * box was at the origin.
   */
  Eigen::Vector3d getNearestPointOnSurfaceRelative(
    const Eigen::Vector3d& point) const;

  Eigen::Vector3d getProjectionOnSideRelative(
    Side side,
    const Eigen::Vector3d& point) const;

  Side getNearestSideRelative(const Eigen::Vector3d& point) const;
  bool isSideNearerRelative(const Eigen::Vector3d& point,
                            Side lhs,
                            Side rhs) const;
  double getDistanceToSideRelative(StaticAxisAlignedBox::Side side,
                                   const Eigen::Vector3d& point) const;

  /**
   * Return the position of the point relative to the center of the
   * box
   */
  Eigen::Vector3d getRelative(const Eigen::Vector3d& point) const;
  /**
   * Return the absolute position of a point positioned relatively
   * to the center of the box.
   */
  Eigen::Vector3d getAbsolute(const Eigen::Vector3d& point) const;

  SideIterator getSideBegin() const;
  SideIterator getSideEnd() const;

  Eigen::Vector3d m_position;
  Eigen::Vector3d m_size;
};

#endif
