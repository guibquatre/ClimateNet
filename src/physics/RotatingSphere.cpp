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
#include "physics/RotatingSphere.hpp"

#include <Eigen/Geometry>

/**
 *  @brief Constructor
 *  @param radial_speed Radial speed in radian.
 */
RotatingSphere::RotatingSphere(const Sphere& sphere,
                               const Eigen::Vector3d& rotation_axe,
                               double radial_speed,
                               double time_before_rotation_start,
                               std::size_t material_identifier)
  : Obstacle(material_identifier), m_sphere(sphere),
    m_rotation_axe(radial_speed * rotation_axe.normalized()),
    m_time_beforme_rotation_start(time_before_rotation_start)
{
}

void
RotatingSphere::update(double step)
{
    m_time_beforme_rotation_start -= step;
}

Mesh
RotatingSphere::getMesh(double number_sector, double number_stack) const noexcept
{
    return m_sphere.getMesh(number_sector, number_stack);
}

/**
 *  @return A transform object that, when applied to any point of the sphere, produce the
 *  position of the point after a call to update with step as argument.
 */
Eigen::Transform<double, 3, Eigen::Affine>
RotatingSphere::getStepUpdate(double step) const
{
    return Eigen::Translation3d(m_sphere.getCenter()) *
           Eigen::AngleAxisd(step * m_rotation_axe.norm(), m_rotation_axe.normalized()) *
           Eigen::Translation3d(-m_sphere.getCenter());
}

/**
 * @brief Check a collision
 * @param[in]  pos            Position to check
 * @param[out] contact_point  Contact point (if valid)
 * @param[out] normal         Normal at the contact point (if valid)
 * @param[out] speed          Speed at the contact point (if valid)
 * @return true if there is a contact (thus contact_point and normal are valid)
 */
bool
RotatingSphere::checkCollision(const Eigen::Vector3d& pos,
                               Eigen::Vector3d& contact_point,
                               Eigen::Vector3d& normal,
                               Eigen::Vector3d& speed) const noexcept
{
    if (!m_sphere.isInside(pos))
    {
        return false;
    }
    m_sphere.project(pos, contact_point, normal);

    if (m_time_beforme_rotation_start < 0)
    {
        Eigen::Vector3d local_contact_point = contact_point - m_sphere.getCenter();
        speed = m_rotation_axe.cross(local_contact_point);
    }
    else
    {
        speed = Eigen::Vector3d::Zero();
    }
    return true;
}
