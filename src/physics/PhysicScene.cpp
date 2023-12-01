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
#include "physics/PhysicScene.hpp"

#include <cassert>
#include <memory>
#include <numeric>
#include <vector>

#include <Eigen/Core>

#include "geometry/geometry_util.hpp"
#include "physics/CollisionConstraint.hpp"
#include "physics/PhysicMesh.hpp"
#include "util.hpp"

PhysicScene::PhysicScene(const FrictionCoefficientTable& friction_coefficients)
  : PhysicScene(FrictionCoefficientTable(friction_coefficients))
{
}

PhysicScene::PhysicScene(FrictionCoefficientTable&& friction_coefficients)
  : m_friction_coefficients(std::move(friction_coefficients))
{
}

PhysicScene::MeshIndexType
PhysicScene::addMesh(const PhysicMesh& mesh)
{
    return addMesh(std::make_shared<PhysicMesh>(mesh));
}

PhysicScene::MeshIndexType
PhysicScene::addMesh(PhysicMesh&& mesh)
{
    return addMesh(std::make_shared<PhysicMesh>(std::move(mesh)));
}

PhysicScene::MeshIndexType
PhysicScene::addMesh(const std::shared_ptr<PhysicMesh>& mesh_ptr)
{
    MeshIndexType mesh_identifier = m_meshes.size();
    m_meshes.push_back(mesh_ptr);
    return mesh_identifier;
}

void
PhysicScene::addObstacle(std::unique_ptr<Obstacle>&& obstacle)
{
    m_obstacles.push_back(std::move(obstacle));
}

const std::vector<PhysicScene::ObstaclePtr>&
PhysicScene::getObstacles() noexcept
{
    return m_obstacles;
}

void
PhysicScene::finalize()
{
    initializeMassMatrices();
}

void
PhysicScene::addConstraint(const std::shared_ptr<AbstractConstraint>& constraint_ptr)
{
    m_user_constraints.push_back(constraint_ptr);
}

const std::vector<std::shared_ptr<AbstractConstraint>>&
PhysicScene::getConstraints()
{
  m_constraints.clear();
  std::size_t current_indice = 0;
  for (auto& mesh_ptr : m_meshes)
  {
    for (const std::shared_ptr<AbstractConstraint>& constraint : mesh_ptr->getConstraints())
    {
      m_constraints.push_back(constraint);
    } // mesh_constraints
  } // mesh
  for (auto& constraint_ptr : m_user_constraints)
  {
    m_constraints.push_back(constraint_ptr);
  } // user_constraints
  return m_constraints;
};



std::vector<CollisionInfo>
PhysicScene::getCollisionsInfo(const Eigen::VectorXd& positions) const
{
    return getCollisionsInfoExceptFor(positions, std::vector<std::size_t>());
}

std::vector<CollisionInfo>
PhysicScene::getCollisionsInfoExceptFor(const Eigen::VectorXd& positions,
                                        const std::vector<std::size_t> exceptions) const
{
    // TODO: Use better data structures to parallelize.
    std::vector<CollisionInfo> result;

    std::vector<ObstaclePtr>::const_iterator obstacle_ptr_iterator;

    const std::size_t vertices_number = getNumberOfVertices();

    std::size_t exception_index = 0;
    std::vector<size_t> index_maping;
    std::vector<Eigen::Vector3d> vertices;
    for (std::size_t position_index = 0; position_index < vertices_number; ++position_index)
    {
        // Check the exceptions, assuming that exceptions is sorted
        while ((exception_index < exceptions.size()) &&
               (exceptions[exception_index] < position_index))
        {
            ++exception_index;
        }
        if ((exception_index < exceptions.size()) && (exceptions[exception_index] == position_index))
        {
            exception_index++;
            continue;
        }

        vertices.push_back(getVector3dBlock(positions, position_index));
        index_maping.push_back(position_index);
    }

    std::vector<std::size_t> colliding_vertices_indices;
    std::vector<Eigen::Vector3d> contact_points;
    std::vector<Eigen::Vector3d> normals;
    std::vector<Eigen::Vector3d> speeds;

    std::size_t colliding_position_index;
    std::size_t colliding_position_material_identifier;
    for (const auto& obstacle_ptr : m_obstacles)
    {
        colliding_vertices_indices.clear();
        contact_points.clear();
        normals.clear();
        speeds.clear();
        obstacle_ptr->checkCollision(
          vertices, colliding_vertices_indices, contact_points, normals, speeds);

        for (std::size_t collision_index = 0; collision_index < contact_points.size();
             ++collision_index)
        {
            colliding_position_index = index_maping[colliding_vertices_indices[collision_index]];
            colliding_position_material_identifier =
              getContainingMesh(colliding_position_index)->getMaterialIdentifier();

            result.push_back(
              CollisionInfo{ colliding_position_index,
                             contact_points[collision_index],
                             normals[collision_index],
                             speeds[collision_index],
                             m_friction_coefficients[colliding_position_material_identifier]
                                                    [obstacle_ptr->getMaterialIdentifier()] });
        }
    }

    return result;
}

std::vector<SelfCollisionInfo>
PhysicScene::getSelfCollisionsInfo(double tol2) const
{
    std::vector<SelfCollisionInfo> result;
    std::vector<SelfCollisionInfo> collisions_infos;

    std::size_t material_identifier;
    for (size_t mId = 0u; mId < m_meshes.size(); ++mId)
    {
        const std::shared_ptr<PhysicMesh> meshPtr = m_meshes[mId];
        material_identifier = meshPtr->getMaterialIdentifier();
        collisions_infos = checkMeshSelfCollisions(*meshPtr, std::sqrt(tol2));
        for (auto& collision_info : collisions_infos)
        {
            collision_info.friction_coefficient = m_friction_coefficients[material_identifier][material_identifier];
        }
        result.insert(result.end(), collisions_infos.begin(), collisions_infos.end());
    }
    return result;
}

const PhysicScene::FrictionCoefficientTable&
PhysicScene::getFrictionCoefficientTable() const noexcept
{
    return m_friction_coefficients;
}

const SparseMatrix&
PhysicScene::getMassMatrix() const
{
    return m_mass_matrix;
}

const SparseMatrix&
PhysicScene::getInverseMassMatrix() const
{
    return m_inverse_mass_matrix;
}

double
PhysicScene::getVertexMass(size_t vId) const noexcept
{
    return m_reduced_mass_matrix.coeff(vId, vId);
}

const SparseMatrix&
PhysicScene::getReducedMassMatrix() const
{
    return m_reduced_mass_matrix;
}

const SparseMatrix&
PhysicScene::getReducedInverseMassMatrix() const
{
    return m_reduced_inverse_mass_matrix;
}

const Eigen::VectorXd&
PhysicScene::getGeneralizedPositions()
{
    if (m_generalized_positions.size() == 0)
    {
        updateGeneralizedPositions();
    }
    return m_generalized_positions;
}
void
PhysicScene::updateTimeDependentObjects(double step)
{
    for (const auto& obstacle_ptr : m_obstacles)
    {
        obstacle_ptr->update(step);
    }
}

void
PhysicScene::updateGeneralizedPositions()
{
    m_generalized_positions = getVectorConcatenation(
      boost::make_transform_iterator(
        m_meshes.begin(), std::bind(&PhysicMesh::getGeneralizedPositions, std::placeholders::_1)),
      boost::make_transform_iterator(
        m_meshes.end(), std::bind(&PhysicMesh::getGeneralizedPositions, std::placeholders::_1)));
}

void
PhysicScene::setGeneralizedPositions(const Eigen::VectorXd& q)
{
    assert(q.size() == getNumberOfDofs());
    m_generalized_positions = q;
    std::size_t current_indice = 0;
    for (auto& mesh_ptr : m_meshes)
    {
        mesh_ptr->setGeneralizedPositions(
          q.segment(current_indice, mesh_ptr->getGeneralizedPositions().size()));
        current_indice += mesh_ptr->getGeneralizedPositions().size();
    }
}

const Eigen::VectorXd&
PhysicScene::getGeneralizedSpeeds()
{
    if (m_generalized_speeds.size() == 0)
    {
        updateGeneralizedSpeeds();
    }
    return m_generalized_speeds;
}

void
PhysicScene::updateGeneralizedSpeeds()
{
    m_generalized_speeds = getVectorConcatenation(
      boost::make_transform_iterator(
        m_meshes.begin(), std::bind(&PhysicMesh::getGeneralizedSpeed, std::placeholders::_1)),
      boost::make_transform_iterator(
        m_meshes.end(), std::bind(&PhysicMesh::getGeneralizedSpeed, std::placeholders::_1)));
}

void
PhysicScene::setGeneralizedSpeeds(const Eigen::VectorXd& speed)
{
    assert(speed.size() == getNumberOfDofs());
    m_generalized_speeds = speed;
    std::size_t current_indice = 0;
    for (auto& mesh_ptr : m_meshes)
    {
        mesh_ptr->setGeneralizedSpeed(
          speed.segment(current_indice, mesh_ptr->getGeneralizedSpeed().size()));
        current_indice += mesh_ptr->getGeneralizedSpeed().size();
    }
}

std::size_t
PhysicScene::getMeshVertexSceneIndex(PhysicScene::MeshIndexType mesh_index,
                                     std::size_t vertex_index) const
{
    return getNumberOfVerticesUntil(mesh_index) + vertex_index;
}

std::shared_ptr<PhysicMesh>
PhysicScene::getMesh(std::size_t index)
{
    return m_meshes[index];
}

std::shared_ptr<const PhysicMesh>
PhysicScene::getMesh(std::size_t index) const
{
    return m_meshes[index];
}

std::size_t
PhysicScene::getNumberOfVertices() const
{
    if (m_generalized_positions.size() == 0)
    {
        return getNumberOfVerticesUntil(m_meshes.size());
    }
    return getVector3dSize(m_generalized_positions);
}

size_t
PhysicScene::getNumberOfDofs() const
{
    if (m_generalized_positions.size() == 0)
    {
        return std::accumulate(getMeshIteratorBegin(),
                               getMeshIteratorEnd(),
                               0,
                               [](std::size_t acc, const PhysicMesh& mesh) {
                                   return mesh.getGeneralizedPositions().size() + acc;
                               });
    }
    return m_generalized_positions.size();
}

void
PhysicScene::initializeMassMatrices()
{
    const std::size_t nDofs = getNumberOfDofs();

    m_mass_matrix = SparseMatrix(nDofs, nDofs);
    m_inverse_mass_matrix = SparseMatrix(nDofs, nDofs);
    const std::size_t nVertices = nDofs / 3u;
    m_reduced_mass_matrix = SparseMatrix(nVertices, nVertices);
    m_reduced_inverse_mass_matrix = SparseMatrix(nVertices, nVertices);

    std::vector<SparseTriplet> triplet_list;
    std::vector<SparseTriplet> triplet_list_inverse;
    std::vector<SparseTriplet> triplet_list_reduced;
    std::vector<SparseTriplet> triplet_list_reduced_inverse;

    size_t vId = 0u;
    for (const std::shared_ptr<PhysicMesh>& mesh_ptr : m_meshes)
    {
        for (size_t mvId = 0u; mvId < mesh_ptr->getNumberOfVertices(); ++mvId)
        {
            const double vMass = mesh_ptr->getVertexMass(mvId);
            triplet_list_reduced.push_back(SparseTriplet(vId, vId, vMass));
            triplet_list_reduced_inverse.push_back(SparseTriplet(vId, vId, 1. / vMass));
            for (size_t cmp = 0u; cmp < 3u; ++cmp)
            {
                const size_t cId = 3u * vId + cmp;
                triplet_list.push_back(SparseTriplet(cId, cId, vMass));
                triplet_list_inverse.push_back(SparseTriplet(cId, cId, 1. / vMass));
            } // cmp
            ++vId;
        } // mvId
    }     // mesh_ptr

    m_mass_matrix.setFromTriplets(triplet_list.begin(), triplet_list.end());
    m_mass_matrix.makeCompressed();

    m_inverse_mass_matrix.setFromTriplets(triplet_list_inverse.begin(), triplet_list_inverse.end());
    m_inverse_mass_matrix.makeCompressed();

    m_reduced_mass_matrix.setFromTriplets(triplet_list_reduced.begin(), triplet_list_reduced.end());
    m_reduced_mass_matrix.makeCompressed();

    m_reduced_inverse_mass_matrix.setFromTriplets(triplet_list_reduced_inverse.begin(),
                                                  triplet_list_reduced_inverse.end());
    m_reduced_inverse_mass_matrix.makeCompressed();
}

std::shared_ptr<PhysicMesh>
PhysicScene::getContainingMesh(std::size_t vertex_index) const
{
    auto begin = m_meshes.begin();
    auto end = m_meshes.end();
    while (begin != end && vertex_index >= (*begin)->getNumberOfVertices())
    {
        vertex_index -= (*begin)->getNumberOfVertices();
        ++begin;
    }
    assert(begin != end);
    return *begin;
}

std::size_t
PhysicScene::getNumberOfVerticesUntil(std::size_t mesh_index) const
{
    auto mesh_iterator_end = getMeshIteratorBegin();
    std::advance(mesh_iterator_end, mesh_index);

    return std::accumulate(
      boost::make_transform_iterator(getMeshIteratorBegin(),
                                     getMeshNumberOfVerticesFunctionObject()),
      boost::make_transform_iterator(mesh_iterator_end, getMeshNumberOfVerticesFunctionObject()),
      0);
}

std::function<std::size_t(const PhysicMesh&)>
PhysicScene::getMeshNumberOfVerticesFunctionObject() const
{
    return std::bind(&PhysicMesh::getNumberOfVertices, std::placeholders::_1);
}

PhysicScene::MeshIterator
PhysicScene::getMeshIteratorBegin() const
{
    return boost::make_transform_iterator(
      m_meshes.begin(), std::bind(&std::shared_ptr<PhysicMesh>::operator*, std::placeholders::_1));
}

PhysicScene::MeshIterator
PhysicScene::getMeshIteratorEnd() const
{
    return boost::make_transform_iterator(
      m_meshes.end(), std::bind(&std::shared_ptr<PhysicMesh>::operator*, std::placeholders::_1));
}
