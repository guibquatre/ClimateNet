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
#ifndef COULOMB_UTIL_HPP
#define COULOMB_UTIL_HPP

#include <Eigen/Core>
#include <vector>

std::vector<Eigen::Vector3d>
getAlartCurnier(const std::vector<Eigen::Vector3d>& velocities,
                const std::vector<Eigen::Vector3d>& forces,
                const std::vector<double> friction_coefficients) noexcept;

std::vector<double>
getAlartCurnierNorms(const std::vector<Eigen::Vector3d>& velocities,
                     const std::vector<Eigen::Vector3d>& forces,
                     const std::vector<double> friction_coefficients) noexcept;

std::vector<double>
getTangentialAlartCurnierNorms(const std::vector<Eigen::Vector3d>& velocities,
                               const std::vector<Eigen::Vector3d>& forces,
                               const std::vector<double> friction_coefficients) noexcept;

std::vector<double>
getNormalAlartCurnierNorms(const std::vector<Eigen::Vector3d>& velocities,
                      const std::vector<Eigen::Vector3d>& forces) noexcept;

double
getAlartCurnierNorm(const Eigen::Vector3d& velocity,
                    const Eigen::Vector3d& force,
                    double friction_coefficient) noexcept;

/**
 *  Compute the Alart Curnier formulation of Signorini Coulomb constraint. The first component of
 *  the velocity and force must be the normal component.
 */
Eigen::Vector3d
getAlartCurnier(const Eigen::Vector3d& velocity,
                const Eigen::Vector3d& force,
                double friction_coefficient) noexcept;

double
getTangentialAlartCurnierNorm(const Eigen::Vector3d& velocity,
                              const Eigen::Vector3d& force,
                              double friction_coefficient) noexcept;

Eigen::Vector2d
getTangentialAlartCurnier(const Eigen::Vector3d& velocity,
                          const Eigen::Vector3d& force,
                          double friction_coefficient) noexcept;

double
getNormalAlartCurnierNorm(const Eigen::Vector3d& velocity, const Eigen::Vector3d& force) noexcept;

double
getNormalAlartCurnier(const Eigen::Vector3d& velocity, const Eigen::Vector3d& force) noexcept;

Eigen::Vector3d::FixedSegmentReturnType<2>::Type
getTangentialComponent(Eigen::Vector3d& v) noexcept;
Eigen::Vector3d::ConstFixedSegmentReturnType<2>::Type
getTangentialComponent(const Eigen::Vector3d& v) noexcept;

double&
getNormalComponent(Eigen::Vector3d& v) noexcept;
double
getNormalComponent(const Eigen::Vector3d& v) noexcept;

#endif

