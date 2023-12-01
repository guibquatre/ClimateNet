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
#ifndef MOVING_MESH_HPP
#define MOVING_MESH_HPP

#include "Obstacle.hpp"
#include "geometry/Mesh.hpp"

//TODO: Change to make it work with timestep greater than the time between main frames.
class MovingMesh : public Obstacle
{
  public:
    MovingMesh(const std::vector<std::string>& meshes_paths,
               const std::vector<double>& meshes_time,
               double collision_tolerance,
               std::size_t material_identifier);

    virtual void update(double step) override final;

  /**
   *    @bried Append the contact point, normals at contact point and speed at contact point to the
   *    given vectors for every given vertex in contact.
   */
  virtual void checkCollision(
          const std::vector<Eigen::Vector3d>& vertices,
          std::vector<size_t>& vertices_indices,
          std::vector<Eigen::Vector3d>& contact_points,
          std::vector<Eigen::Vector3d>& normals,
          std::vector<Eigen::Vector3d>& speeds) const noexcept override final;

    /**
     * @brief Check a collision
     * @param[in]  pos            Position to check
     * @param[out] contact_point  Contact point (if valid)
     * @param[out] normal         Normal at the contact point (if valid)
     * @param[out] speed          Speed of the contact point (if valid)
     * @return true if there is a contact (thus contact_point and normal are valid)
     */
    bool checkCollision(const Eigen::Vector3d& pos,
                        Eigen::Vector3d& contact_point,
                        Eigen::Vector3d& normal,
                        Eigen::Vector3d& speed) const noexcept override final;

  private:
    /**
     * @brief From Christer Ericson -- Real-Time Collision Detection (p141)
     *
     * @param p     Vertex to project
     * @param a     Triangle 1st vertex
     * @param b     Triangle 2nd vertex
     * @param c     Triangle 3rd vertex
     * @param tol2  Tolerance for the collision
     *
     * @return The barycentric coordinate of the closest point to p inside (abc)
     */
    static Eigen::Vector3d closestPointToTriangle(const Eigen::Vector3d& p,
                                                  const Eigen::Matrix3d triangle,
                                                  const double tol2);

    bool isInMovement() const noexcept;

    Eigen::Vector3d getPointSpeed(const Eigen::Vector3d& barycentric_coordinate,
                                  const std::array<size_t, 3>& triangle_indices) const noexcept;

    Mesh& getPreviousMainFrame() noexcept;
    Mesh& getNextMainFrame() noexcept;
    const Mesh& getPreviousMainFrame() const noexcept;
    const Mesh& getNextMainFrame() const noexcept;
    std::size_t getPreviousMainFrameIndex() const noexcept;
    std::size_t getNextMainFrameIndex() const noexcept;
    std::size_t getNumberMainFrames() const noexcept;
    double getTimeBetweenPreviousAndNextMainFrame() const noexcept;
    double getPreviousMainFrameTime() const noexcept;
    double getNextMainFrameTime() const noexcept;
    double getPreviousMainframeFilenameIndex() const noexcept;
    double getNextMainFrameFilenameIndex() const noexcept;

    void updateCurrentMainFrame();
    void updateCurrentMesh() noexcept;
    Mesh loadMesh(std::size_t filename_index) const;

    std::vector<std::string> m_main_frames_filenames;
    std::vector<double> m_main_frames_times;
    std::array<Mesh, 2> m_current_main_frames;
    Mesh m_current_mesh;
    double m_current_time;
    std::size_t m_previous_main_frame_index;
    std::size_t m_next_main_frame_filename_index;
    double m_collision_tolerance_squared;
};

#endif
