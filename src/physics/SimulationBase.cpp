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
#include "physics/SimulationBase.hpp"

#include "timer.hpp"
#include "physics/io.hpp"

#include "omp_utils.hpp"
#include "util.hpp"

#include "geometry/geometry_util.hpp"
#include <numeric>
#include <algorithm>

/**
 * @brief Dummy filter function
 * @param 
 * @param 
 * @return true
 */
constexpr bool
isAnything(std::size_t, const Eigen::Vector3d&)
{
    return true;
}



SimulationBase::SimulationBase(
  PhysicScene& scene,
  double time_step,
  std::size_t iteration_number,
  const SimulationBase::ExternalForce& external_force,
  double air_damping,
  bool handle_self_collision,
  double self_collision_tolerance) :
  m_nVertices(scene.getNumberOfVertices()),
  m_nDofs(3u * scene.getNumberOfVertices()), 
  m_time_step(time_step), m_current_time(0),
  m_iteration_number(iteration_number),
  m_damping_coefficient(std::max(air_damping, 0.)),
  m_handle_self_collision(handle_self_collision),
  m_self_collision_tol2(self_collision_tolerance * self_collision_tolerance),
  m_scene(scene),
  m_external_force_fun(external_force),
  m_constant_constraints(scene.getConstraints()),
  m_relative_velocities_filter(&isAnything),
  m_current_step_number_of_collisions(iteration_number),
  m_current_step_error(iteration_number),
  m_current_step_duration(iteration_number)
{
  m_current_external_force.setZero(m_nDofs);

  std::vector<SparseTriplet> triplet_list;
  for (const auto& constraint_ptr : m_constant_constraints)
  {
    constraint_ptr->addConstraintMatrix(triplet_list);
  }
  m_A = SparseMatrix(triplet_list.back().row() + 1, m_nVertices);
  m_A.setFromTriplets(triplet_list.begin(), triplet_list.end());
  m_A.makeCompressed();

  m_projections = Eigen::MatrixXd::Zero(3, triplet_list.back().row() + 1);

  triplet_list.clear();
  for (size_t vId = 0u; vId < m_nVertices; ++vId) {
    triplet_list.push_back(SparseTriplet(vId, vId, 0.));
  }
  m_contact_matrix_old = SparseMatrix(m_nVertices, m_nVertices);
  m_contact_matrix_new = SparseMatrix(m_nVertices, m_nVertices);
  m_contact_matrix_old.setFromTriplets(triplet_list.begin(),
                                       triplet_list.end());
  m_contact_matrix_new.setFromTriplets(triplet_list.begin(),
                                       triplet_list.end());
  m_contact_matrix_old.makeCompressed();
  m_contact_matrix_new.makeCompressed();

  m_rhs = Eigen::MatrixXd::Zero(3, m_nVertices);

  m_damping = Eigen::VectorXd::Zero(m_nVertices);
  m_t_n = Eigen::VectorXd::Zero(m_nDofs);
  m_current_next_position = Eigen::VectorXd::Zero(m_nDofs);
}


void
SimulationBase::step()
{
  
  // Compute the constant terms
  initializeStep();
  // Log data
  double solver_error;

  TIMER_START(step);
  
  // Detect and add collisions
  baseUpdateColisions();

  
  // PD iterations
  for (std::size_t i = 0; i < m_iteration_number; ++i)
  {

    TIMER_START(iteration);
    
    // Recompute the estimations
    TIMER_START(rhs);
    buildIteration();
    const double duration_rhs = TIMER_DURATION(rhs, microseconds);
    m_rhs_times.push_back(duration_rhs);
#ifdef TIMER_PRINT
    std::cout << "## Local step (RHS computation) : "
              << duration_rhs << " µs" << std::endl;
#endif // TIMER_PRINT
    
    // Global step
    TIMER_START(global);
    solver_error = solve();
    const double duration_global = TIMER_DURATION(global, microseconds);
    m_global_times.push_back(duration_global);
#ifdef TIMER_PRINT
    std::cout << "## Global step (linear system)  : "
              << duration_global << " µs" << std::endl;
#endif // TIMER_PRINT
    //std::cout << "Error: " << solver_error << std::endl;
		
    // Log
    const double duration_iteration = TIMER_DURATION(iteration, microseconds);
    m_iteration_times.push_back(duration_iteration);
#ifdef TIMER_PRINT
    std::cout << "#### Iteration " << i << " : "
              << duration_iteration << " µs" << std::endl;
#endif // TIMER_PRINT
    updateNumberCollisions(i, m_collisions_infos.size());
    updateErrors(i, solver_error);
    updateDurations(i, TIMER_DELTA(step) );
    this->updateIterationData();

  }
  
  // Update and end the step
  updateScene();
  // IO
  updateRelativeVelocities();
  m_current_time += m_time_step;
  std::cout << m_current_time << std::endl;

  
  const double duration_step = TIMER_DURATION(step, microseconds);
  m_step_times.push_back(duration_step);
#ifdef TIMER_PRINT
  std::cout << "######## Total timestep : "
            << duration_step << " µs" << std::endl;
#endif // TIMER_PRINT
}




void
SimulationBase::setRelativeVelocitiesFilter(
  const RelativeVelocitiesFilter& filter) noexcept
{
    setRelativeVelocitiesFilter(RelativeVelocitiesFilter(filter));
}

void
SimulationBase::setRelativeVelocitiesFilter(
  RelativeVelocitiesFilter&& filter) noexcept
{
    m_relative_velocities_filter = std::move(filter);
}

// TODO: Factorize get<Something>RelativeVelocityNorm
double
SimulationBase::getMaxRelativeVelocityNorm() const noexcept
{
    std::function<double(const Eigen::Vector3d&)> norm_function =
      std::mem_fn(&Eigen::Vector3d::norm);

    auto begin = boost::make_transform_iterator(
      m_current_step_relative_velocities.begin(), norm_function);
    auto end = boost::make_transform_iterator(
      m_current_step_relative_velocities.end(), norm_function);

    auto result_it = std::max_element(begin, end);
    if (result_it == end)
    {
        return -1;
    }
    else
    {
        return *result_it;
    }
}

double
SimulationBase::getMeanRelativeVelocityNorm() const noexcept
{
    std::function<double(const Eigen::Vector3d&)> norm_function =
      std::mem_fn(&Eigen::Vector3d::norm);

    auto begin = boost::make_transform_iterator(
      m_current_step_relative_velocities.begin(), norm_function);
    auto end = boost::make_transform_iterator(
      m_current_step_relative_velocities.end(), norm_function);

    if (begin == end)
    {
        return -1;
    }

    return std::accumulate(begin, end, 0.) / (double)
           m_current_step_relative_velocities.size();
}

const std::vector<double>&
SimulationBase::getLastStepErrors() const noexcept
{
    return m_current_step_error;
}

const std::vector<SimulationBase::IterationDuration>&
SimulationBase::getLastStepDurations() const noexcept
{
    return m_current_step_duration;
}

const std::vector<std::size_t>&
SimulationBase::
getLastStepNumberCollisions() const noexcept
{
    return m_current_step_number_of_collisions;
}

const Eigen::VectorXd&
SimulationBase::getCurrentNextPosition() const noexcept
{
  return m_current_next_position;
}

const Eigen::VectorXd&
SimulationBase::getCurrentNextSpeed() const noexcept
{
  return m_current_next_speed;
}

void
SimulationBase::computeCurrentExternalForce() noexcept {
  
  const Eigen::VectorXd &generalized_position = m_scene.getGeneralizedPositions();
  const SparseMatrix &mass_mat = m_scene.getMassMatrix();
  
  Eigen::Vector3d force_i;
#pragma omp parallel for private(force_i)
  for (std::size_t vId = 0; vId < m_scene.getNumberOfVertices(); ++vId)
  {
    
    const auto mass_mat_i = mass_mat.block(3u * vId, 3u * vId, 3, 3);
    force_i = m_external_force_fun(mass_mat_i,
                                   getVector3dBlock(generalized_position, vId),
                                   m_current_time);
    for (size_t cmp = 0u; cmp < 3u; ++cmp)
    {
#pragma omp atomic write
      m_current_external_force[3u * vId + cmp] = force_i[cmp];
    } // cmp
  } // i
}
const Eigen::VectorXd&
SimulationBase::getCurrentExternalForce() const noexcept
{
  return m_current_external_force;
}

void 
SimulationBase::computeNextSpeedUnderNoConstraint(
  const Eigen::VectorXd& current_external_forces) noexcept
{
  m_t_n = m_scene.getGeneralizedSpeeds()
    + m_time_step * m_scene.getInverseMassMatrix() *
    current_external_forces;
}
const Eigen::VectorXd&
SimulationBase::getNextSpeedUnderNoConstraint() const noexcept
{
  return m_t_n;
}

const SparseMatrix &
SimulationBase::getA() const noexcept
{
  return m_A;
}

void
SimulationBase::computeProjections(const Eigen::VectorXd& positions,
                                   const Eigen::VectorXd& speeds)
{
#pragma omp parallel for
  for (size_t cId = 0u; cId < m_constant_constraints.size(); ++cId)
  {
    m_constant_constraints[cId]->addConstraintProjection(positions, m_projections);
  }
}


const Eigen::Matrix3Xd&
SimulationBase::getProjections() const noexcept
{
  return m_projections;
}


void
SimulationBase::baseUpdateColisions() noexcept
{
  TIMER_START(collision_detection);
  
  m_collisions_infos =
    //  m_scene.getCollisionsInfo(getCurrentNextPosition());
    m_scene.getCollisionsInfo(m_scene.getGeneralizedPositions());
  
  if (m_collisions_infos.size() > 0) 
  {
    m_collision_numbers.push_back(m_collisions_infos.size());
  }

  m_damping.setConstant(m_damping_coefficient);
  if (m_damping_coefficient > 0.)
  {
#pragma omp parallel
    for (size_t cId = 0u; cId < m_collisions_infos.size(); ++cId)
    {
      m_damping[m_collisions_infos[cId].vertex_index] = 0.;
    } // cId
  }
  
  const double duration_collision_detection =
    TIMER_DURATION(collision_detection, microseconds);
  m_collision_detection_times.push_back(duration_collision_detection);
#ifdef TIMER_PRINT
  std::cout << "# Collision detection          : "
            << duration_collision_detection
            << " µs" << std::endl;
  std::cout << "Number of collision            : "
            << m_collisions_infos.size() << std::endl;
#endif // TIMER_PRINT



  if (m_handle_self_collision)
  {
    TIMER_START(self_collision_detection);
    
    m_self_collisions_infos =
      m_scene.getSelfCollisionsInfo(m_self_collision_tol2);
    if (m_self_collisions_infos.size() > 0) 
    {
      m_self_collision_numbers.push_back(m_self_collisions_infos.size());
    }

    if (m_damping_coefficient > 0.)
    {
#pragma omp parallel
      for (size_t scId = 0u; scId < m_self_collisions_infos.size(); ++scId)
      {
        m_damping[m_self_collisions_infos[scId].vertex_index] = 0.;
        // other vertex
        const Eigen::Vector3d &alpha = m_self_collisions_infos[scId].barycentric_coordinates;
        const std::array<size_t, 3> &nId = m_self_collisions_infos[scId].face_indices;
        const size_t ovId =
          (alpha[0] > alpha[1]) ?
          ((alpha[0] > alpha[2]) ? nId[0] : nId[2]) :
          ((alpha[1] > alpha[2]) ? nId[1] : nId[2]);
        m_damping[ovId] = 0.;
      } // vId
    }

    
    const double duration_self_collision_detection =
      TIMER_DURATION(self_collision_detection, microseconds);
    m_self_collision_detection_times.push_back(duration_self_collision_detection);
#ifdef TIMER_PRINT
    std::cout << "# Self-collision detection     : "
              << duration_self_collision_detection
              << " µs" << std::endl;
    std::cout << "Number of self collision       : " << m_self_collisions_infos.size() << std::endl;
#endif // TIMER_PRINT

    
  }
  
  updateCollision();
}


const SparseMatrix&
SimulationBase::getLhs() const noexcept
{
  return m_lhs;
}

const Eigen::Matrix<double, 3, -1, Eigen::RowMajor>&
SimulationBase::getRhs() const noexcept
{
  return m_rhs;
}

void
SimulationBase::updateNumberCollisions(std::size_t iteration,
                                       std::size_t collisions_number) noexcept
{
  m_current_step_number_of_collisions[iteration] = collisions_number;
}

void
SimulationBase::updateRelativeVelocities() noexcept
{
  m_current_step_relative_velocities.clear();
  const Eigen::VectorXd &velocities = m_scene.getGeneralizedSpeeds();
  const Eigen::VectorXd &positions = m_scene.getGeneralizedPositions();
  for (const CollisionInfo& collision_info : m_collisions_infos)
  {
    if (!m_relative_velocities_filter(
          collision_info.vertex_index,
          getVector3dBlock(positions, collision_info.vertex_index)))
    {
      continue;
    }

    // All obstacle are static, the relative velocities is therefore
    // the same as the velocities.
    m_current_step_relative_velocities.emplace_back(
      getChangeOfBasisIntoOrthonormalDirectBasis(collision_info.normal) *
      getVector3dBlock(velocities, collision_info.vertex_index));
  }
}

void
SimulationBase::updateErrors(std::size_t iteration, double error) noexcept
{
  m_current_step_error[iteration] = error;
}

void
SimulationBase::updateDurations(
  std::size_t iteration,
  std::chrono::high_resolution_clock::duration duration) noexcept
{
    m_current_step_duration[iteration] = duration;
}


void
SimulationBase::exportStats(std::ofstream &of) const noexcept
{
  computeAndWriteStats("Iteration time (µs)", of,
                       m_iteration_times);
  computeAndWriteStats("Collision detection time (µs)", of,
                       m_collision_detection_times);
  if (m_handle_self_collision)
  {
    computeAndWriteStats("Number of collisions", of,
                         m_collision_numbers);
    computeAndWriteStats("Self collision detection time (µs)", of,
                         m_self_collision_detection_times);
  }
  computeAndWriteStats("Number of self collisions", of,
                       m_self_collision_numbers);
  computeAndWriteStats("Solver time (µs)", of,
                       m_global_times);
  computeAndWriteStats("Full RHS computation time (µs)", of,
                       m_rhs_times);
  computeAndWriteStats("Step time (µs)", of, m_step_times);
}
