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
#include "physics/PhysicMesh.hpp"

#include "boost/iterator/counting_iterator.hpp"
#include "physics/BendingConstraint.hpp"
#include "util.hpp"

PhysicMesh::PhysicMesh(const std::vector<Eigen::Vector3d>& positions,
                       const std::vector<tinyobj::index_t>& triangles,
                       const PhysicParameters& physic_parameters,
                       std::size_t material_identifier) :
  Mesh(positions, triangles),
  MaterialObject(material_identifier),
  m_physic_parameters(physic_parameters),
  m_speed(Eigen::VectorXd::Zero(positions.size() * 3))
{
  initializeMasses();
  initializeConstraints();
}

const std::vector<std::shared_ptr<AbstractConstraint>>&
PhysicMesh::getConstraints() const
{
  return m_constraints;
}

const Eigen::VectorXd&
PhysicMesh::getGeneralizedSpeed() const
{
  return m_speed;
}

Eigen::Vector3d
PhysicMesh::getVertexSpeed(size_t index) const
{
  return getVector3dBlock(getGeneralizedSpeed(), index);
}

void
PhysicMesh::setGeneralizedSpeed(const Eigen::VectorXd& speed)
{
  m_speed = speed;
}

double
PhysicMesh::getVertexMass(size_t vertex_index) const
{
  return m_masses[vertex_index];
}


void PhysicMesh::initializeConstraints()
{
  std::transform(
    getEdgeBegin(),
    getEdgeEnd(),
    std::back_inserter(m_constraints),
    std::bind(&PhysicMesh::getEdgeConstraint, this, std::placeholders::_1));
  std::transform(
    getInteriorVerticesBegin(),
    getInteriorVerticesEnd(),
    std::back_inserter(m_constraints),
    std::bind(&PhysicMesh::getVertexConstraint, this, std::placeholders::_1));
}

std::unique_ptr<AbstractConstraint>
PhysicMesh::getVertexConstraint(std::size_t vertex_index) const
{
  auto vertices_around_vertex = getVerticesAroundVertexRange(vertex_index);
  std::vector<std::size_t> vertex_indices;
  vertex_indices.push_back(vertex_index);
  vertex_indices.insert(vertex_indices.end(),
                        boost::begin(vertices_around_vertex),
                        boost::end(vertices_around_vertex));

  return std::make_unique<BendingConstraint>(
    vertex_indices,
    getVertex(vertex_index),
    std::vector<Eigen::Vector3d>(
      boost::make_transform_iterator(
        boost::begin(vertices_around_vertex),
        std::bind(&Mesh::getVertex, this, std::placeholders::_1)),
      boost::make_transform_iterator(
        boost::end(vertices_around_vertex),
        std::bind(&Mesh::getVertex, this, std::placeholders::_1))),
    m_physic_parameters.bend);
}

std::unique_ptr<AbstractConstraint>
PhysicMesh::getEdgeConstraint(const Edge& edge) const
{
  double rest_length = getEdgeLength(edge);
  return std::make_unique<SpringConstraint>(
    edge.vertex1_index, edge.vertex2_index, m_physic_parameters.stretch, rest_length);
}

void
PhysicMesh::initializeMasses()
{
  m_masses.resize(getNumberOfVertices());
  for (std::size_t vId = 0; vId < m_masses.size(); ++vId)
  {
    auto triangle_around_vertex_mass_range =
      getTriangleAroundVertexMassRange(vId);
    m_masses[vId] = std::accumulate(boost::begin(triangle_around_vertex_mass_range),
                                    boost::end(triangle_around_vertex_mass_range),
                                    0.) / 3;
  }
}

boost::iterator_range<
  boost::transform_iterator<std::function<double(const PhysicMesh::Triangle&)>,
                            PhysicMesh::TriangleIterator>>
  PhysicMesh::getTriangleAroundVertexMassRange(std::size_t vertex_index) const
{
  return boost::make_iterator_range(
    boost::make_transform_iterator(getTriangleAroundVertexBegin(vertex_index),
                                   getTriangleMassFunctionObject()),
    boost::make_transform_iterator(getTriangleAroundVertexEnd(vertex_index),
                                   getTriangleMassFunctionObject()));
}

std::function<double(const PhysicMesh::Triangle&)>
PhysicMesh::getTriangleMassFunctionObject() const
{
  return std::bind(&PhysicMesh::getTriangleMass, this, std::placeholders::_1);
}

double
PhysicMesh::getTriangleMass(const Triangle& triangle) const
{
  return m_physic_parameters.area_density * getTriangleArea(triangle);
}
