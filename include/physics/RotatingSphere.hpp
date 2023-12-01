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
#ifndef ROTATING_SPHERE_HPP
#define ROTATING_SPHERE_HPP

#include "geometry/Sphere.hpp"
#include "physics/Obstacle.hpp"

// TODO: Factorize StaticSphere and RotatingSphere
class RotatingSphere : public Obstacle
{
  public:
    /**
     *  @brief Constructor
     *  @param radial_speed Radial speed in radian.
     */
    RotatingSphere(const Sphere& sphere,
                   const Eigen::Vector3d& rotation_axe,
                   double radial_speed,
                   double time_before_rotation_start,
                   std::size_t material_identifier);

    virtual void update(double step) override final;

    Mesh getMesh(double number_sector, double number_stack) const noexcept;

    /**
     *  @return A transform object that, when applied to any point of the sphere, produce the
     *  position of the point after a call to update with step as argument.
     */
    Eigen::Transform<double, 3, Eigen::Affine> getStepUpdate(double step) const;

    /**
     * @brief Check a collision
     * @param[in]  pos            Position to check
     * @param[out] contact_point  Contact point (if valid)
     * @param[out] normal         Normal at the contact point (if valid)
     * @param[out] speed          Speed at the contact point (if valid)
     * @return true if there is a contact (thus contact_point and normal are valid)
     */
    virtual bool checkCollision(const Eigen::Vector3d& pos,
                                Eigen::Vector3d& contact_point,
                                Eigen::Vector3d& normal,
                                Eigen::Vector3d& speed) const noexcept override final;

  private:
    Sphere m_sphere;
    double m_time_beforme_rotation_start;
    /// @brief Rotation vector, i.e. normalized rotation axe times rotation speed.
    Eigen::Vector3d m_rotation_axe;
};

#endif
