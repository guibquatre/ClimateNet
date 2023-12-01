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
#include "physics/MovingMesh.hpp"
#include "geometry/geometry_util.hpp"
#include "geometry/io.hpp"

#include <Eigen/Geometry>

// TODO: factorize StaticMesh and MovingMesh

MovingMesh::MovingMesh(const std::vector<std::string>& main_frames_filenames,
                       const std::vector<double>& main_frames_times,
                       double collision_tolerance,
                       std::size_t material_identifier)
  : Obstacle(material_identifier), m_main_frames_filenames(main_frames_filenames),
    m_main_frames_times(main_frames_times), m_current_main_frames{ loadMesh(0), loadMesh(1) },
    m_current_mesh(loadMesh(0)), m_current_time(0), m_previous_main_frame_index(0),
    m_next_main_frame_filename_index(1),
    m_collision_tolerance_squared(collision_tolerance * collision_tolerance)
{
    assert(m_main_frames_filenames.size() == m_main_frames_times.size());
    // For the obstacle to move, at least two meshes are necessary
    assert(m_main_frames_filenames.size() >= 2);
}

void
MovingMesh::update(double step)
{
    m_current_time += step;

    updateCurrentMainFrame();
    updateCurrentMesh();
}

void
MovingMesh::updateCurrentMainFrame()
{
    if (getNextMainFrameTime() < m_current_time &&
        getNextMainFrameFilenameIndex() + 1 < getNumberMainFrames())
    {
        ++m_next_main_frame_filename_index;
        m_previous_main_frame_index = (m_previous_main_frame_index + 1) % 2;

        std::string next_main_frame_filename =
          m_main_frames_filenames[m_next_main_frame_filename_index];
        m_current_main_frames[getNextMainFrameIndex()].setGeneralizedPositions(
          read_obj_generalized_positions(next_main_frame_filename));
    }
}

void
MovingMesh::updateCurrentMesh() noexcept
{
    // Don't need to update if before first main frame or after second main frame
    if (!isInMovement())
    {
        return;
    }
    double last_main_frame_time = m_main_frames_times[m_next_main_frame_filename_index - 1];
    double next_main_frame_time = m_main_frames_times[m_next_main_frame_filename_index];
    double interpolation_coeff =
      (m_current_time - last_main_frame_time) / (next_main_frame_time - last_main_frame_time);

    Eigen::VectorXd current_mesh_new_generalized_position =
      getPreviousMainFrame().getGeneralizedPositions() * (1 - interpolation_coeff) +
      getNextMainFrame().getGeneralizedPositions() * interpolation_coeff;

    m_current_mesh.setGeneralizedPositions(std::move(current_mesh_new_generalized_position));
}

Mesh&
MovingMesh::getPreviousMainFrame() noexcept
{
    return m_current_main_frames[getPreviousMainFrameIndex()];
}

Mesh&
MovingMesh::getNextMainFrame() noexcept
{
    return m_current_main_frames[getNextMainFrameIndex()];
}

const Mesh&
MovingMesh::getPreviousMainFrame() const noexcept
{
    return m_current_main_frames[getPreviousMainFrameIndex()];
}

const Mesh&
MovingMesh::getNextMainFrame() const noexcept
{
    return m_current_main_frames[getNextMainFrameIndex()];
}

std::size_t
MovingMesh::getNextMainFrameIndex() const noexcept
{
    return (getPreviousMainFrameIndex() + 1) % 2;
}

std::size_t
MovingMesh::getPreviousMainFrameIndex() const noexcept
{
    return m_previous_main_frame_index;
}

double
MovingMesh::getTimeBetweenPreviousAndNextMainFrame() const noexcept
{
    return getNextMainFrameTime() - getPreviousMainFrameTime();
}

double
MovingMesh::getPreviousMainFrameTime() const noexcept
{
    return m_main_frames_times[getPreviousMainframeFilenameIndex()];
}

double
MovingMesh::getNextMainFrameTime() const noexcept
{
    return m_main_frames_times[getNextMainFrameFilenameIndex()];
}

double
MovingMesh::getPreviousMainframeFilenameIndex() const noexcept
{
    return m_next_main_frame_filename_index - 1;
}

double
MovingMesh::getNextMainFrameFilenameIndex() const noexcept
{
    return m_next_main_frame_filename_index;
}

Mesh
MovingMesh::loadMesh(std::size_t filename_index) const
{
    return read_obj(m_main_frames_filenames[filename_index]);
}

/**
 *    @bried Append the contact point, normals at contact point and speed at contact point to the
 *    given vectors for every given vertex in contact.
 */
void
MovingMesh::checkCollision(const std::vector<Eigen::Vector3d>& vertices,
                           std::vector<size_t>& vertices_indices,
                           std::vector<Eigen::Vector3d>& contact_points,
                           std::vector<Eigen::Vector3d>& normals,
                           std::vector<Eigen::Vector3d>& speeds) const noexcept
{
    std::vector<Eigen::Vector3d> barycentric_coordinates;
    std::vector<Mesh::Triangle> triangles;

    checkMeshCollision(vertices,
                       m_current_mesh,
                       std::sqrt(m_collision_tolerance_squared),
                       vertices_indices,
                       contact_points,
                       normals,
                       &barycentric_coordinates,
                       &triangles);

    for (std::size_t collision_index = 0; collision_index < barycentric_coordinates.size();
         ++collision_index)
    {
      if (!isInMovement())
      {
        speeds.push_back(Eigen::Vector3d::Zero());
      }
      else {
        speeds.push_back(getPointSpeed(barycentric_coordinates[collision_index],
                                       triangles[collision_index].vertex_indices));
      }
    }
}

bool
MovingMesh::checkCollision(const Eigen::Vector3d& pos,
                           Eigen::Vector3d& contact_point,
                           Eigen::Vector3d& normal,
                           Eigen::Vector3d& speed) const noexcept
{
    Eigen::Vector3d contact_point_barycentric_coordinate;
    std::array<size_t, 3> contact_triangle_vertices_indices;

    if (!checkMeshCollision(pos,
                            m_current_mesh,
                            std::sqrt(m_collision_tolerance_squared),
                            contact_point,
                            normal,
                            &contact_point_barycentric_coordinate,
                            nullptr,
                            &contact_triangle_vertices_indices))
    {
        return false;
    }

    // Check of we are after the first main frame or before the first main frame.
    if (!isInMovement())
    {
        speed = Eigen::Vector3d::Zero();
    }
    else
    {
        speed =
          getPointSpeed(contact_point_barycentric_coordinate, contact_triangle_vertices_indices);
    }

    return true;
}

Eigen::Vector3d
MovingMesh::getPointSpeed(const Eigen::Vector3d& barycentric_coordinate,
                          const std::array<size_t, 3>& triangle_indices) const noexcept
{
    Eigen::Matrix3d previous_triangle_positions;
    Eigen::Matrix3d next_triangle_positions;
    for (std::size_t component_index = 0; component_index < 3; ++component_index)
    {
        std::size_t closest_vertex_index = triangle_indices[component_index];
        previous_triangle_positions.col(component_index) =
          getPreviousMainFrame().getVertex(closest_vertex_index);
        next_triangle_positions.col(component_index) =
          getNextMainFrame().getVertex(closest_vertex_index);
    }
    Eigen::Vector3d next_position = next_triangle_positions * barycentric_coordinate;
    Eigen::Vector3d previous_position = previous_triangle_positions * barycentric_coordinate;
    return (next_position - previous_position) / getTimeBetweenPreviousAndNextMainFrame();
}

/**
 * @return True if we are before the first main frame or after the last main frame
 */
bool
MovingMesh::isInMovement() const noexcept
{
    return m_current_time >= getPreviousMainFrameTime() && m_current_time <= getNextMainFrameTime();
}

std::size_t
MovingMesh::getNumberMainFrames() const noexcept
{
    return m_main_frames_filenames.size();
}
