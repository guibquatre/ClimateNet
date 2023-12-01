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
#ifndef STATIC_OBSTACLE_HPP
#define STATIC_OBSTACLE_HPP

#include "Obstacle.hpp"

class StaticObstacle : public Obstacle
{
  public:
    using Obstacle::Obstacle;

    virtual void update(double step) override;

    /**
     *    @bried Append the contact point, normals at contact point and speed at contact point to
     * the given vectors for every given vertex in contact.
     */
    virtual void checkCollision(const std::vector<Eigen::Vector3d>& vertices,
                                std::vector<size_t>& vertices_indices,
                                std::vector<Eigen::Vector3d>& contact_points,
                                std::vector<Eigen::Vector3d>& normals) const noexcept;

    /**
     *    @bried Append the contact point, normals at contact point and speed at contact point to
     * the given vectors for every given vertex in contact.
     */
    virtual void checkCollision(const std::vector<Eigen::Vector3d>& vertices,
                                std::vector<size_t>& vertices_indices,
                                std::vector<Eigen::Vector3d>& contact_points,
                                std::vector<Eigen::Vector3d>& normals,
                                std::vector<Eigen::Vector3d>& speeds) const noexcept override final;

    /**
     * @brief Check a collision
     * @param[in]  pos            Position to check
     * @param[out] contact_point  Contact point (if valid)
     * @param[out] normal         Normal at the contact point (if valid)
     * @return true if there is a contact (thus contact_point and normal are valid)
     */
    virtual bool checkCollision(const Eigen::Vector3d& pos,
                                Eigen::Vector3d& contact_point,
                                Eigen::Vector3d& normal) const noexcept = 0;

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
                                Eigen::Vector3d& speed) const noexcept override final;
};

#endif // STATIC_OBSTACLE
