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
#include "physics/SimulationFrictionEstimation.hpp"
#include <Eigen/Dense>
#include <unordered_set>

#include "physics/coulomb_util.hpp"
#include "physics/io.hpp"
#include "timer.hpp"

#include "physics/CollisionConstraint.hpp"
#include "util.hpp"

SimulationFrictionEstimation::
SimulationFrictionEstimation(PhysicScene& scene,
                             double time_step,
                             std::size_t iteration_number,
                             const ExternalForce& external_force,
                             double air_damping,
                             bool handle_self_collision,
                             double self_collision_tolerance)
  : SimulationSpeed(scene,
                    time_step,
                    iteration_number,
                    external_force,
                    air_damping,
                    handle_self_collision,
                    self_collision_tolerance),
    m_self_collision_graph(m_scene.getNumberOfVertices())
{
    // Allocate the required storage
    if (m_handle_self_collision)
    {
        m_current_index = 0u;
        m_next_index = 1u;
        for (unsigned int i = 0u; i < 2u; ++i)
        {
            m_contact_forces[i].setZero(3, m_nVertices);
            m_self_contact_forces[i].setZero(3, m_nVertices);
            m_self_contact_repercusion_forces[i].setZero(3, m_nVertices);
            m_alpha[i].setZero(3, m_nVertices);
        } // i
    }     // m_handle_self_collision

}

void
SimulationFrictionEstimation::updateScene() noexcept
{
  // Just removing the damping from the parent class
  m_scene.setGeneralizedPositions(getCurrentNextPosition());
  m_scene.setGeneralizedSpeeds(m_current_next_speed);
  m_scene.updateTimeDependentObjects(m_time_step);

  // Cleaning the forces for the next step
  if (m_handle_self_collision)
  {
    m_current_index = 0u;
    m_next_index = 1u;
    for (unsigned int i = 0u; i < 2u; ++i)
    {
      m_contact_forces[i].setZero();
      m_self_contact_forces[i].setZero();
      m_self_contact_repercusion_forces[i].setZero();
      m_alpha[i].setZero();
    }
  }

  
}

const Eigen::Vector3d test1(1., 0., 0.);
const Eigen::Vector3d test2(0., 1., 0.);
void
computeBasis(const Eigen::Vector3d& normal, Eigen::Matrix3d& basis)
{
    Eigen::Vector3d tangent = normal.cross(test1);
    if (tangent.norm() < 1.e-15)
    {
        tangent = normal.cross(test2);
    }
    tangent.normalize();
    const Eigen::Vector3d bitangent = normal.cross(tangent).normalized();

    basis.col(0) = normal;
    basis.col(1) = tangent;
    basis.col(2) = bitangent;
}

void
SimulationFrictionEstimation::updateCollision() noexcept
{
    // Remove the collision gestion of the parent
    // Build the rotations

    // Contact with an external object
    const unsigned int nbCollisions = m_collisions_infos.size();
    m_local_contact_basis.resize(nbCollisions);

    // Self contact
    const unsigned int nbSelfCollisions = m_self_collisions_infos.size();
    m_local_self_contact_basis.resize(nbSelfCollisions);

    m_remember_self_contact_forces = Eigen::Matrix3Xd::Zero(3, nbSelfCollisions);

#pragma omp parallel
    {
#pragma omp for nowait
        for (unsigned int cId = 0u; cId < nbCollisions; ++cId)
        {
            const CollisionInfo& collision_info = m_collisions_infos[cId];
            const Eigen::Vector3d& normal = collision_info.normal;
            computeBasis(normal, m_local_contact_basis[cId]);
        } // cId

#pragma omp for
        for (unsigned int scId = 0u; scId < nbSelfCollisions; ++scId)
        {
            const CollisionInfo& self_collision_info = m_self_collisions_infos[scId];
            const Eigen::Vector3d& normal = self_collision_info.normal;
            computeBasis(normal, m_local_self_contact_basis[scId]);
        } // scId
    }     // omp parallel

    if (m_handle_self_collision && !m_self_collisions_infos.empty())
    {
        TIMER_START(collision_ordering)
        computeCollisionComputationOrder();
        const double collision_ordering_duration = TIMER_DURATION(collision_ordering, microseconds);
#ifdef TIMER_PRINT
        std::cout << "# Collision Ordering: " << collision_ordering_duration << "µs" << std::endl;
#endif // TIMER_PRINT
        m_self_collision_ordering_times.push_back(collision_ordering_duration);
    }
}

void 
SimulationFrictionEstimation::buildIteration() noexcept 
{
    computeRhs(getCurrentNextSpeed(),
               getNextSpeedUnderNoConstraint());
}

void
SimulationFrictionEstimation::computeCollisionComputationOrder() noexcept
{
    computeSelfCollisionGraph();
    m_collision_computation_order.clear();
    std::vector<bool> visited(m_self_collisions_infos.size(), false);
    std::vector<std::size_t> starting_vertices;

    for (const auto& collision_info : m_collisions_infos)
    {
        starting_vertices.push_back(collision_info.vertex_index);
    }
    fillCollisionComputationOrder(starting_vertices, visited, 0u);

    for (std::size_t vertex_index = 0; vertex_index < m_scene.getNumberOfVertices(); ++vertex_index)
    {
        if (vertexIsVisited(vertex_index, visited))
        {
            continue;
        }

        if (m_self_collision_graph[vertex_index].size() == 1u)
        {
            starting_vertices.clear();
            starting_vertices.push_back(vertex_index);
            fillCollisionComputationOrder(starting_vertices, visited, 0u);
        }
    }

    for (std::size_t vertex_index = 0; vertex_index < m_scene.getNumberOfVertices(); ++vertex_index)
    {
        if (vertexIsVisited(vertex_index, visited))
        {
            continue;
        }

        if (m_self_collision_graph[vertex_index].size() > 1u)
        {
            starting_vertices.clear();
            starting_vertices.push_back(vertex_index);
            fillCollisionComputationOrder(starting_vertices, visited, 0u);
        }
    }
}

void
SimulationFrictionEstimation::computeSelfCollisionGraph() noexcept
{
    for (std::size_t vertex_index = 0; vertex_index < m_scene.getNumberOfVertices(); ++vertex_index)
    {
        m_self_collision_graph[vertex_index].clear();
    }

    // Graph Computation
    for (std::size_t self_collision_info_index = 0;
         self_collision_info_index < m_self_collisions_infos.size();
         ++self_collision_info_index)
    {
        const SelfCollisionInfo& self_collision_info =
          m_self_collisions_infos[self_collision_info_index];

        std::size_t target = getSelfCollisionContactPointIndex(self_collision_info);

        std::size_t source = self_collision_info.vertex_index;

        m_self_collision_graph[source].push_back(
          SelfCollisionGraphNeighboor{ target, self_collision_info_index });
        m_self_collision_graph[target].push_back(
          SelfCollisionGraphNeighboor{ source, self_collision_info_index });
    }
}

std::size_t
SimulationFrictionEstimation::getSelfCollisionContactPointIndex(
  const SelfCollisionInfo& self_collision_info) const noexcept
{
    const Eigen::Vector3d& alpha = self_collision_info.barycentric_coordinates;
    return self_collision_info
      .face_indices[(alpha[0] > alpha[1]) ? ((alpha[0] > alpha[2]) ? 0 : 2)
                                          : ((alpha[1] > alpha[2]) ? 1 : 2)];
}

void
SimulationFrictionEstimation::fillCollisionComputationOrder(const std::vector<std::size_t>& starting_vertices,
                                              std::vector<bool>& visited,
                                              std::size_t initial_computation_index) noexcept
{
    std::vector<std::size_t> visited_last = starting_vertices;
    std::vector<std::size_t> visiting;

    std::size_t computation_index = initial_computation_index;
    while (!visited_last.empty())
    {
        if (m_collision_computation_order.size() <= computation_index)
        {
            m_collision_computation_order.emplace_back();
        }

        for (std::size_t vertex_index : visited_last)
        {
            for (const SelfCollisionGraphNeighboor& neighboor :
                 m_self_collision_graph[vertex_index])
            {
                if (visited[neighboor.collision_index])
                {
                    continue;
                }

                visiting.push_back(neighboor.index);
                visited[neighboor.collision_index] = true;
                m_collision_computation_order[computation_index].push_back(
                  neighboor.collision_index);
            }
        }
        visited_last = std::move(visiting);
        visiting.clear();
        ++computation_index;
    }
}

bool
SimulationFrictionEstimation::vertexIsVisited(std::size_t vertex_index, const std::vector<bool>& visited) const
  noexcept
{
    for (const auto& neighboor : m_self_collision_graph[vertex_index])
    {
        if (!visited[neighboor.collision_index])
        {
            return false;
        }
    }
    return true;
}

// TODO: factorize with the stuff in util

#define BLOCK(m, id) ((m).col((id)))
#define LVAL(m, id, cmp) ((m).coeffRef((cmp), (id)))

struct SelfForceToAdd
{
    size_t id_plus;
    size_t id_minus;
    Eigen::Vector3d force;
};

void
SimulationFrictionEstimation::computeRhs(const Eigen::VectorXd& current_next_speed,
                                         const Eigen::VectorXd& next_speed_under_no_constraints) noexcept
{
    // Regular RHS without the friction forces
    TIMER_START(rhs);
    SimulationSpeed::computeRhs(current_next_speed, next_speed_under_no_constraints);
    // Missing term is dt * r and - LHS w
    //                   ^^
    const double duration_base_rhs = TIMER_DURATION(rhs, microseconds);
    m_rhs_base_times.push_back(duration_base_rhs);
#ifdef TIMER_PRINT
    std::cout << "# Base rhs : " << duration_base_rhs << " µs" << std::endl;
#endif // TIMER_PRINT

    const unsigned int nbCollisions = m_collisions_infos.size();
    const unsigned int nbSelfCollisions = m_self_collisions_infos.size();
    m_collisions_local_forces.resize(nbCollisions);
    m_self_collisions_local_forces.resize(nbSelfCollisions);

    // Abort if there is no collisions
    if ((nbCollisions == 0) && (!m_handle_self_collision || (nbSelfCollisions == 0)))
    {
        return;
    }

    TIMER_START(friction);

    // Reconstruct the forces without any friction force

    Eigen::Matrix3Xd forces = m_rhs;
    const auto v = Eigen::MatrixXd::Map(current_next_speed.data(), 3, m_nVertices);
    
#pragma omp parallel for
    for (size_t cmp = 0u; cmp < 3u; ++cmp)
    {
        forces.row(cmp) -=
          // std::pow(m_time_step, 2) * (getATA() * v.row(cmp).transpose()).transpose();
          //// ! better perf !
          std::pow(m_time_step, 2) * v.row(cmp) * getATA();
    }

    // Estimated friction force
#pragma omp parallel for
    for (unsigned int cId = 0u; cId < nbCollisions; ++cId)
    {
        // Data
        const CollisionInfo& collision_info = m_collisions_infos[cId];
        const size_t vId = collision_info.vertex_index;
        const Eigen::Matrix3d& local_basis = m_local_contact_basis[cId];
        const double mu = collision_info.friction_coefficient;

        // Converting the rhs in the local basis
        // Contact force also sees the previous self collision forces
        Eigen::Vector3d forceLoc;
        if (!m_handle_self_collision)
        {
            forceLoc = local_basis.transpose() *
                       (BLOCK(forces, vId) - m_scene.getVertexMass(vId) * collision_info.speed);
        }
        else
        {
            forceLoc = local_basis.transpose() *
                       (BLOCK(forces, vId) +
                        BLOCK(m_self_contact_forces[m_current_index], vId)
                        //+ BLOCK(m_self_contact_repercusion_forces[m_current_index], vId)
                        - m_scene.getVertexMass(vId) * collision_info.speed);
        }

        // Estimating the reaction
        if (forceLoc[0] > 0.)
        {
            // take-off
            continue;
        }
        Eigen::Vector3d r(0., 0., 0.);
        // rN must prevent the penetration
        r[0] = -forceLoc[0];

        // rT try to prevent the sliding
        // Sticking
        r[1] = -forceLoc[1];
        r[2] = -forceLoc[2];

        const double rT_norm = sqrt(r[1] * r[1] + r[2] * r[2]);
        if (rT_norm > mu * r[0])
        {
            // but gets stuck at the border of the cone
            // Sliding
            r[1] *= mu * r[0] / rT_norm;
            r[2] *= mu * r[0] / rT_norm;
        }

        m_collisions_local_forces[cId] = r;

        // Converting r in the global frame
        r = local_basis * r;

        if ((!m_handle_self_collision) || (nbSelfCollisions == 0))
        {
            // Adding it directly to the rhs
            for (unsigned int cmp = 0u; cmp < 3u; ++cmp)
            {
                //#pragma omp atomic update Not needed : max 1 contact per vertex
                LVAL(m_rhs, vId, cmp) += r[cmp];
            } // cmp
        }
        if (m_handle_self_collision)
        {
            // Storing it
            for (unsigned int cmp = 0u; cmp < 3u; ++cmp)
            {
                LVAL(m_contact_forces[m_next_index], vId, cmp) = r[cmp];
            } // cmp
        }     // self

    } // cId

    const double duration_friction = TIMER_DURATION(friction, microseconds);
    m_friction_times.push_back(duration_friction);
#ifdef TIMER_PRINT
    std::cout << "# Rhs friction : " << duration_friction << " µs" << std::endl;
#endif // TIMER_PRINT

    if ((m_handle_self_collision) && (nbSelfCollisions > 0))
    {

        TIMER_START(self_friction);

        // Estimated self friction force
        for (size_t level = 0u; level < m_collision_computation_order.size(); ++level)
        {
            const std::vector<size_t>& collisions_level_ids = m_collision_computation_order[level];

            std::vector<SelfForceToAdd> forces_to_add;
            forces_to_add.resize(collisions_level_ids.size());

#pragma omp parallel for
            for (unsigned int i = 0u; i < collisions_level_ids.size(); ++i)
            {
                const size_t scId = collisions_level_ids[i];
                // Data
                const SelfCollisionInfo& self_collision_info = m_self_collisions_infos[scId];
                const size_t vId = self_collision_info.vertex_index;
                const std::array<size_t, 3> fId = self_collision_info.face_indices;
                const Eigen::Matrix3d& local_basis = m_local_self_contact_basis[scId];
                const Eigen::Vector3d& alpha = self_collision_info.barycentric_coordinates;
                const double mu = self_collision_info.friction_coefficient;

                // Self contact force can see the previous contact forces
                // and the previous repercusion forces
                // BUT not the one it spawned
                const double mA = m_scene.getVertexMass(vId);
                // Assign to the closest
                /*
                            const double mB = alpha[0] * m_scene.getVertexMass(fId[0]) +
                                              alpha[1] * m_scene.getVertexMass(fId[1]) +
                                              alpha[2] * m_scene.getVertexMass(fId[2]);
                */
                const size_t idB = (alpha[0] > alpha[1])
                                     ? ((alpha[0] > alpha[2]) ? fId[0] : fId[2])
                                     : ((alpha[1] > alpha[2]) ? fId[1] : fId[2]);
                const double mB = m_scene.getVertexMass(idB);

                const Eigen::Vector3d fA =
                  BLOCK(forces, vId) +
                  BLOCK(m_contact_forces[m_next_index], vId)
                  //+ BLOCK(m_self_contact_repercusion_forces[m_current_index], vId)
                  + BLOCK(m_self_contact_forces[m_current_index], vId) // levels >=
                  + BLOCK(m_self_contact_forces[m_next_index], vId)    // levels <
                  - m_remember_self_contact_forces.col(scId);
                /*
                Eigen::Vector3d fB(0., 0., 0.);
                for (unsigned int i = 0u; i < 3u; ++i)
                {
                    fB += alpha[i] * mB / m_scene.getVertexMass(fId[i]) *
                          (BLOCK(forces, fId[i]) + BLOCK(m_contact_forces[m_current_index], fId[i])
                + BLOCK(m_self_contact_repercusion_forces[m_current_index], fId[i])
                           // Removing its own repercusion force
                           + m_alpha[m_current_index](vId, i) *
                               BLOCK(m_self_contact_forces[m_current_index], vId));
                }
                */
                const Eigen::Vector3d fB =
                  BLOCK(forces, idB) +
                  BLOCK(m_contact_forces[m_next_index], idB)
                  //+ BLOCK(m_self_contact_repercusion_forces[m_current_index], idB)
                  + BLOCK(m_self_contact_forces[m_current_index], idB) // levels >=
                  + BLOCK(m_self_contact_forces[m_next_index], idB)    // levels <
                  + m_remember_self_contact_forces.col(scId);

                const Eigen::Vector3d forceLoc =
                  local_basis.transpose() * ((1. / mA) * fA - (1. / mB) * fB);

                // Estimating the reaction
                if (forceLoc[0] > 0.)
                {
                    // take-off
                    forces_to_add[i].id_plus = vId;
                    forces_to_add[i].id_minus = idB;
                    forces_to_add[i].force.setZero();
                    continue;
                }
                // rN must prevent the penetration
                // rT try to prevent the sliding
                Eigen::Vector3d r = -forceLoc;
                const double rT_norm = sqrt(r[1] * r[1] + r[2] * r[2]);
                if (rT_norm > mu * r[0])
                {
                    // but gets stuck at the border of the cone
                    // Sliding
                    r[1] *= mu * r[0] / rT_norm;
                    r[2] *= mu * r[0] / rT_norm;
                }

                // Renormalizing r
                r = ((mA * mB) / (mA + mB)) * r;
                m_self_collisions_local_forces[scId] = r;
                // Converting r in the global frame
                r = local_basis * r;

                forces_to_add[i].id_plus = vId;
                forces_to_add[i].id_minus = idB;
                forces_to_add[i].force = r;
                
            
            } // scId

#pragma omp parallel for
            for (size_t i = 0u; i < collisions_level_ids.size(); ++i)
            {
                const size_t scId = collisions_level_ids[i];
                const Eigen::Vector3d prev_force = m_remember_self_contact_forces.col(scId);
                const SelfForceToAdd& new_force = forces_to_add[i];

                for (size_t cmp = 0u; cmp < 3u; ++cmp)
                {
                    // Remove from current
#pragma omp atomic update
                    LVAL(m_self_contact_forces[m_current_index], new_force.id_plus, cmp) -=
                      prev_force[cmp];
#pragma omp atomic update
                    LVAL(m_self_contact_forces[m_current_index], new_force.id_minus, cmp) +=
                      prev_force[cmp];

#pragma omp atomic update
                    LVAL(m_self_contact_forces[m_next_index], new_force.id_plus, cmp) +=
                      new_force.force[cmp];
#pragma omp atomic update
                    LVAL(m_self_contact_forces[m_next_index], new_force.id_minus, cmp) -=
                      new_force.force[cmp];

#pragma omp atomic write
                    m_remember_self_contact_forces(cmp, scId) = new_force.force[cmp];
                } // cmp
            }     // scId

        } // level

        // Add all the forces to the RHS
        m_rhs += m_contact_forces[m_next_index] + m_self_contact_forces[m_next_index]
          //+ m_self_contact_repercusion_forces[m_next_index]
          ;

        const double duration_self_friction = TIMER_DURATION(self_friction, microseconds);
        m_self_friction_times.push_back(duration_self_friction);
#ifdef TIMER_PRINT
        std::cout << "# Rhs self friction : " << duration_self_friction << " µs" << std::endl;
#endif // TIMER_PRINT
    }

    // Move on next step
    if (m_handle_self_collision)
    {
        m_current_index = m_next_index;
        m_next_index = (m_next_index) ? 0u : 1u;
        m_contact_forces[m_next_index].setZero();
        m_self_contact_forces[m_next_index].setZero();
        // m_self_contact_repercusion_forces[m_next_index].setZero();
        m_alpha[m_next_index].setZero();
    }
}

void
SimulationFrictionEstimation::updateAlartCurnierNorms() noexcept
{
    updateCollisionsLocalVelocities();

    m_alart_curnier_norms = getAlartCurnierNorms(
      m_collisions_local_velocities, m_collisions_local_forces, getCollisionFrictionCoefficients());
    m_tangential_alart_curnier_norms = getTangentialAlartCurnierNorms(
      m_collisions_local_velocities, m_collisions_local_forces, getCollisionFrictionCoefficients());
    m_normal_alart_curnier =
      getNormalAlartCurnierNorms(m_collisions_local_velocities, m_collisions_local_forces);

    m_self_alart_curnier_norms = getAlartCurnierNorms(m_self_collisions_local_velocities,
                                                      m_self_collisions_local_forces,
                                                      getSelfCollisionFrictionCoefficients());
    m_self_tangential_alart_curnier_norms =
      getTangentialAlartCurnierNorms(m_self_collisions_local_velocities,
                                     m_self_collisions_local_forces,
                                     getSelfCollisionFrictionCoefficients());
    m_self_normal_alart_curnier = getNormalAlartCurnierNorms(m_self_collisions_local_velocities,
                                                             m_self_collisions_local_forces);
}

void
SimulationFrictionEstimation::updateCollisionsLocalVelocities() noexcept
{
    /*
    updateLocalVelocities(
      getCollisionVerticesIndices(), m_local_contact_basis, m_collisions_local_velocities);

    updateLocalVelocities(getSelfCollisionVerticesIndices(),
                          m_local_self_contact_basis,
                          m_self_collisions_local_velocities);
   
   /*/
   // Dirty fix
   const size_t nbCollisions = m_collisions_infos.size();
   m_collisions_local_velocities.resize(nbCollisions);
   for (unsigned int cId = 0u; cId < nbCollisions; ++cId) {
       const auto& info = m_collisions_infos[cId];
       m_collisions_local_velocities[cId] = 
            m_local_contact_basis[cId].transpose() *
               (getVector3dBlock(m_current_next_speed, info.vertex_index)
                 - info.speed);
   } // cId

   const size_t nbSelfCollisions = m_self_collisions_infos.size();
   m_self_collisions_local_velocities.resize(nbSelfCollisions);
   for (unsigned int scId = 0u; scId < nbSelfCollisions; ++scId) {
       const auto &info = m_self_collisions_infos[scId];
       const auto &alpha = info.barycentric_coordinates;
       const size_t nId = 
          (alpha[0] > alpha[1]) ? 
             ( (alpha[0] > alpha[2]) ? info.face_indices[0] : info.face_indices[2] ) :
             ( (alpha[1] > alpha[2]) ? info.face_indices[1] : info.face_indices[2] );
        m_self_collisions_local_velocities[scId] = 
            m_local_self_contact_basis[scId].transpose() *
                (getVector3dBlock(m_current_next_speed, info.vertex_index)
                  - getVector3dBlock(m_current_next_speed, nId));

   } // scId
   /* */
}

// TODO: There should be a way to factorize this efficientely. (Maybe use pointer to member?)
std::vector<std::size_t>
SimulationFrictionEstimation::getSelfCollisionVerticesIndices() const noexcept
{
    std::vector<size_t> result;
    result.reserve(m_self_collisions_infos.size());
    for (const auto& self_collision_info : m_self_collisions_infos)
    {
        result.push_back(self_collision_info.vertex_index);
    }
    return result;
}

std::vector<std::size_t>
SimulationFrictionEstimation::getCollisionVerticesIndices() const noexcept
{
    std::vector<size_t> result;
    result.reserve(m_collisions_infos.size());
    for (const auto& collision_info : m_collisions_infos)
    {
        result.push_back(collision_info.vertex_index);
    }
    return result;
}

std::vector<double>
SimulationFrictionEstimation::getSelfCollisionFrictionCoefficients()
  const noexcept
{
    std::vector<double> result;
    result.reserve(m_self_collisions_infos.size());
    for (const auto& self_collision_info : m_self_collisions_infos)
    {
        result.push_back(self_collision_info.friction_coefficient);
    }
    return result;
}

std::vector<double>
SimulationFrictionEstimation::getCollisionFrictionCoefficients() const noexcept
{
    std::vector<double> result;
    result.reserve(m_collisions_infos.size());
    for (const auto& collision_info : m_collisions_infos)
    {
        result.push_back(collision_info.friction_coefficient);
    }
    return result;
}

void
SimulationFrictionEstimation::updateLocalVelocities(
  const std::vector<std::size_t>& vertices_indices,
  const std::vector<Eigen::Matrix3d>& local_basis,
  std::vector<Eigen::Vector3d>& local_velocities) noexcept
{
    local_velocities.resize(vertices_indices.size());
    for (std::size_t collision_index = 0; collision_index < vertices_indices.size();
         ++collision_index)
    {
        local_velocities[collision_index] =
          local_basis[collision_index].transpose()
          * getVector3dBlock(m_current_next_speed, vertices_indices[collision_index]);
    }
}

void
SimulationFrictionEstimation::updateIterationData() noexcept
{
    updateAlartCurnierNorms();
#ifdef PRINT_ALART_CURNIER_ERROR
    if (!m_alart_curnier_norms.empty())
    {
        std::cout << "TangentialAlartCurnier ";
        std::copy(m_tangential_alart_curnier_norms.begin(),
                  m_tangential_alart_curnier_norms.end(),
                  std::ostream_iterator<double>(std::cout, " "));
        std::cout << std::endl;
        std::cout << "NormalAlartCurnier ";
        std::copy(m_normal_alart_curnier.begin(),
                  m_normal_alart_curnier.end(),
                  std::ostream_iterator<double>(std::cout, " "));
        std::cout << std::endl;
    }

    if (!m_self_alart_curnier_norms.empty())
    {
        std::cout << "SelfTangentialAlartCurnier ";
        std::copy(m_self_tangential_alart_curnier_norms.begin(),
                  m_self_tangential_alart_curnier_norms.end(),
                  std::ostream_iterator<double>(std::cout, " "));
        std::cout << std::endl;
        std::cout << "SelfNormalAlartCurnier ";
        std::copy(m_self_normal_alart_curnier.begin(),
                  m_self_normal_alart_curnier.end(),
                  std::ostream_iterator<double>(std::cout, " "));
        std::cout << std::endl;
    }
#endif
}

void
SimulationFrictionEstimation::exportStats(std::ofstream& of) const noexcept
{
    computeAndWriteStats("LHS computation time (µs)", of, m_lhs_times);
    computeAndWriteStats("Cholesky computation time (µs)", of, m_cholesky_times);

    SimulationBase::exportStats(of);

    computeAndWriteStats("RHS base computation time (µs)", of, m_rhs_base_times);
    computeAndWriteStats("RHS friction overhead (µs)", of, m_friction_times);
    if (m_handle_self_collision)
    {
        computeAndWriteStats("Self collision time (µs)", of, m_self_friction_times);
        computeAndWriteStats("Self collision ordering time (µs)", of, m_self_collision_ordering_times);
    }

}
