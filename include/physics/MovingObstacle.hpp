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
#ifndef MOVING_OBSTACLE_HPP
#define MOVING_OBSTACLE_HPP

#include "Obstacle.hpp"

class MovingObstacle : public Obstacle
{
  public:
    using Obstacle::Obstacle;

    virtual void update(double step) = 0;

    /**
     * @brief Check a collision
     * @param[in]  pos            Position to check
     * @param[out] contact_point  Contact point (if valid)
     * @param[out] normal         Normal at the contact point (if valid)
     * @param[out] speed          Speed of the contact point (if valid)
     * @return true if there is a contact (thus contact_point and normal are valid)
     */
    virtual bool checkCollision(const Eigen::Vector3d& pos,
                                Eigen::Vector3d& contact_point,
                                Eigen::Vector3d& normal,
                                Eigen::Vector3d& speed) const noexcept = 0;
};

#endif // MOVING_OBSTACLE
