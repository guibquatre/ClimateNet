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
#include "physics/coulomb_util.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <iostream>

// TODO: Find a ternary transform in the STL or any other library to factorize this.
std::vector<Eigen::Vector3d>
getAlartCurnier(const std::vector<Eigen::Vector3d>& velocities,
                const std::vector<Eigen::Vector3d>& forces,
                const std::vector<double> friction_coefficients) noexcept
{
    std::vector<Eigen::Vector3d> result(velocities.size());
    for (std::size_t index = 0; index < velocities.size(); ++index)
    {
        result[index] =
          getAlartCurnier(velocities[index], forces[index], friction_coefficients[index]);
    }
    return result;
}

std::vector<double>
getAlartCurnierNorms(const std::vector<Eigen::Vector3d>& velocities,
                     const std::vector<Eigen::Vector3d>& forces,
                     const std::vector<double> friction_coefficients) noexcept
{
    std::vector<double> result(velocities.size());
    for (std::size_t index = 0; index < velocities.size(); ++index)
    {
        result[index] =
          getAlartCurnierNorm(velocities[index], forces[index], friction_coefficients[index]);
    }
    return result;
}

std::vector<double>
getTangentialAlartCurnierNorms(const std::vector<Eigen::Vector3d>& velocities,
                               const std::vector<Eigen::Vector3d>& forces,
                               const std::vector<double> friction_coefficients) noexcept
{
    std::vector<double> result(velocities.size());
    for (std::size_t index = 0; index < velocities.size(); ++index)
    {
        result[index] = getTangentialAlartCurnierNorm(
          velocities[index], forces[index], friction_coefficients[index]);
    }
    return result;
}

std::vector<double>
getNormalAlartCurnierNorms(const std::vector<Eigen::Vector3d>& velocities,
                           const std::vector<Eigen::Vector3d>& forces) noexcept
{
    std::vector<double> result(velocities.size());
    for (std::size_t index = 0; index < velocities.size(); ++index)
    {
        result[index] = getNormalAlartCurnierNorm(velocities[index], forces[index]);
    }
    return result;
}

std::vector<double>
getNormalAlartCurnier(const std::vector<Eigen::Vector3d>& velocities,
                      const std::vector<Eigen::Vector3d>& forces) noexcept
{
    std::vector<double> result(velocities.size());
    for (std::size_t index = 0; index < velocities.size(); ++index)
    {
        result[index] = getNormalAlartCurnier(velocities[index], forces[index]);
    }
    return result;
}

double
getAlartCurnierNorm(const Eigen::Vector3d& velocity,
                    const Eigen::Vector3d& force,
                    double friction_coefficient) noexcept
{
    return getAlartCurnier(velocity, force, friction_coefficient).norm();
    /// (velocity.norm() + force.norm());
}

Eigen::Vector3d
getAlartCurnier(const Eigen::Vector3d& velocity,
                const Eigen::Vector3d& force,
                double friction_coefficient) noexcept
{
    Eigen::Vector3d result;
    getNormalComponent(result) = getNormalAlartCurnier(velocity, force);
    getTangentialComponent(result) =
      getTangentialAlartCurnier(velocity, force, friction_coefficient);

    return result;
}

double
getTangentialAlartCurnierNorm(const Eigen::Vector3d& velocity,
                              const Eigen::Vector3d& force,
                              double friction_coefficient) noexcept
{
    return getTangentialAlartCurnier(velocity, force, friction_coefficient).norm();
}

Eigen::Vector2d
getTangentialAlartCurnier(const Eigen::Vector3d& velocity,
                          const Eigen::Vector3d& force,
                          double friction_coefficient) noexcept
{
    const Eigen::Vector2d force_velocity_tangential_difference =
      getTangentialComponent(force) - getTangentialComponent(velocity);
    const double radius 
        = std::max(0., 
                   friction_coefficient * getNormalComponent(force) 
                     / force_velocity_tangential_difference.norm());
    return force_velocity_tangential_difference
             * std::min(1., radius)
           - getTangentialComponent(force);
}

double
getNormalAlartCurnierNorm(const Eigen::Vector3d& velocity, const Eigen::Vector3d& force) noexcept
{
    return std::abs(getNormalAlartCurnier(velocity, force));
}

double
getNormalAlartCurnier(const Eigen::Vector3d& velocity, const Eigen::Vector3d& force) noexcept
{

    // std::cout << "NormalComponent: " << getNormalComponent(force) << ", "
    //          << getNormalComponent(velocity) << std::endl;
    return std::max(0., getNormalComponent(force) - getNormalComponent(velocity))
           - getNormalComponent(force);
}

Eigen::Vector3d::FixedSegmentReturnType<2>::Type
getTangentialComponent(Eigen::Vector3d& v) noexcept
{
    return v.segment<2>(1);
}

Eigen::Vector3d::ConstFixedSegmentReturnType<2>::Type
getTangentialComponent(const Eigen::Vector3d& v) noexcept
{
    return v.segment<2>(1);
}

double&
getNormalComponent(Eigen::Vector3d& v) noexcept
{
    return v.x();
}
double
getNormalComponent(const Eigen::Vector3d& v) noexcept
{
    return v.x();
}
