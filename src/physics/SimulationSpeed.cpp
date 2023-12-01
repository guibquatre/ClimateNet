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
#include "physics/SimulationSpeed.hpp"

#include "timer.hpp"
#include "physics/io.hpp"

#include "physics/CollisionConstraint.hpp"
#include "util.hpp"


SimulationSpeed::SimulationSpeed(PhysicScene& scene,
                                 double time_step,
                                 std::size_t iteration_number,
                                 const ExternalForce& external_force,
                                 double air_damping,
                                 bool handle_self_collision,
                                 double self_collision_tolerance) :
  SimulationBase(scene, time_step, iteration_number,
                 external_force, air_damping,
                 handle_self_collision, self_collision_tolerance)
{
  // Constant over the whole simulation
  // LHS for the constraints

  TIMER_START(lhs);
  computeLhs();
  const double duration_lhs = TIMER_DURATION(lhs, microseconds);
  m_lhs_times.push_back(duration_lhs);
#ifdef TIMER_PRINT
  std::cout << "######## Initial LHS computation        : "
            << duration_lhs << " µs" << std::endl;
#endif // TIMER_PRINT
  
  TIMER_START(cholesky)
  m_solver.compute(getLhs());
  const double duration_cholesky = TIMER_DURATION(cholesky, microseconds);
  m_cholesky_times.push_back(duration_cholesky);
#ifdef TIMER_PRINT
  std::cout << "######## Initial Cholesky factorization : " 
            << duration_cholesky << " µs" << std::endl;
#endif // TIMER_PRINT
}

void
SimulationSpeed::initializeStep() noexcept
{
  // Constant over the time-step
  // External forces
  computeCurrentExternalForce();
  // Predicted speed w/o constraints
  computeNextSpeedUnderNoConstraint(getCurrentExternalForce());
  
  // Initial guess
  m_current_next_speed = getNextSpeedUnderNoConstraint();
  
  // ...
  computeCurrentNextPosition();
}


void
SimulationSpeed::buildIteration() noexcept
{
  // Rebuild the base RHS
  computeRhs(getCurrentNextSpeed(),
             getNextSpeedUnderNoConstraint());

 // Add the collisions

  const Eigen::VectorXd &generalized_positions
    = m_scene.getGeneralizedPositions();
#pragma omp parallel for  
  for (size_t cId = 0u; cId < m_collisions_infos.size(); ++cId)
  {
    const CollisionInfo& collision_info =
      m_collisions_infos[cId];

    // Rhs
    const Eigen::Vector3d proj =
      CollisionConstraint::getProjection(getCurrentNextPosition(),
                                         collision_info.vertex_index,
                                         collision_info.contact_point,
                                         collision_info.normal);
    
    for (size_t cmp = 0u; cmp < 3u; ++cmp)
    {
#pragma omp atomic update
      m_rhs(cmp, collision_info.vertex_index) +=
        m_time_step * COLLISION_CONSTRAINT_WEIGHT
        * (proj(cmp)
           - generalized_positions(3u * collision_info.vertex_index + cmp));
        
    } // cmp
   } // cId
}

void
SimulationSpeed::updateCollision() noexcept
{
  TIMER_START(collision);

  // Compute the new
  m_contact_matrix_new *= 0;

  const double t2 = std::pow(m_time_step, 2);
  
//#pragma omp parallel for  
  for (size_t cId = 0u; cId < m_collisions_infos.size(); ++cId)
  {
    const CollisionInfo& collision_info =
      m_collisions_infos[cId];
    
    // Lhs
#pragma omp atomic update
    m_contact_matrix_new.
      coeffRef(collision_info.vertex_index,
               collision_info.vertex_index) +=
      t2 * COLLISION_CONSTRAINT_WEIGHT;

  } // cId
  
  
  const bool matrix_changed =
    ((m_contact_matrix_new - m_contact_matrix_old).norm() >= 1.e-10);
  
  if (matrix_changed) {
    
    m_lhs += m_contact_matrix_new - m_contact_matrix_old;
    m_contact_matrix_old = m_contact_matrix_new;

    const double duration_collision_update =
      TIMER_DURATION(collision, microseconds);
    m_lhs_update_times.push_back(duration_collision_update);
#ifdef TIMER_PRINT
    std::cout << "# Collision - update system    : "
              << duration_collision_update << " µs" << std::endl;
#endif // TIMER_PRINT
    // Collisions modified the LHS, need to refactorize
    // NB: we could test if the LHS didn't change, but it's tedious...
    // We have to check if the same points are contacting with the same normal
    TIMER_START(cholesky);
    m_solver.factorize(getLhs());
    const double duration_cholesky = TIMER_DURATION(cholesky, microseconds);
    m_cholesky_times.push_back(duration_cholesky);
#ifdef TIMER_PRINT
    std::cout << "# Cholesky recomputation       : "
              << duration_cholesky << " µs" << std::endl;
#endif // TIMER_PRINT
  }
  
}

double
SimulationSpeed::solve() noexcept
{
  // Change the view
  auto current_next_speed =
    Eigen::MatrixXd::Map(m_current_next_speed.data(),
                         3, m_nVertices);

  // Keep the pragma
#pragma omp parallel for
  for (size_t cmp = 0u; cmp < 3u; ++cmp) {
    current_next_speed.row(cmp) =
      m_solver.solve(m_rhs.row(cmp).transpose()).transpose();  
  }
  computeCurrentNextPosition();
  return 0.;
}

void
SimulationSpeed::updateScene() noexcept
{
  // Dampen the velocities
  //m_current_next_speed = getCollisionDampingCoefficients() * m_current_next_speed;
  // Dirty fix
  /*
  for (unsigned int cId = 0u; cId < m_collisions_infos.size(); ++cId) {
    const unsigned int vId = m_collisions_infos[cId].vertex_index;
    const Eigen::Vector3d &normal = m_collisions_infos[cId].normal;
    Eigen::Vector3d speed = m_current_next_speed.segment<3>(3u * vId);
    double normalSpeed = speed.dot(normal);
    speed -= normalSpeed * normal;
    speed *= (1. - 6.5e-1);
    speed += normalSpeed * normal;
    m_current_next_speed.segment<3>(3u * vId) = speed;
  } // cId
  */
  // Update the scene
  m_scene.setGeneralizedPositions(getCurrentNextPosition());
  m_scene.setGeneralizedSpeeds(m_current_next_speed);
}

void
SimulationSpeed::computeCurrentNextPosition() noexcept
{
  m_current_next_position =
    m_scene.getGeneralizedPositions()
    + m_time_step * m_current_next_speed;
}

void
SimulationSpeed::computeCurrentNextSpeed() noexcept
{}

const SparseMatrix&
SimulationSpeed::getATA() const noexcept
{
  return m_ATA;
}

void
SimulationSpeed::computeLhs() noexcept
{
  m_ATA = getA().transpose() * getA();
  m_lhs = m_scene.getReducedMassMatrix();
  m_lhs += std::pow(m_time_step, 2) * getATA();
}

void
SimulationSpeed::computeRhs(
  const Eigen::VectorXd& current_next_speed,
  const Eigen::VectorXd& next_speed_under_no_constraints) noexcept
{
  // Change the view
  
  const auto positions =
    Eigen::MatrixXd::Map(m_scene.getGeneralizedPositions().data(),
                         3, m_nVertices);
  const auto speed =
    Eigen::MatrixXd::Map(next_speed_under_no_constraints.data(),
                         3, m_nVertices);
  
  const auto nextSpeed =
    Eigen::MatrixXd::Map(current_next_speed.data(),
                         3, m_nVertices);

  // Constraints (velocity) term
  computeProjections(getCurrentNextPosition(), m_current_next_speed);
  
  /// !? Needed for eigen optimization
  const Eigen::Matrix<double, 3, -1, Eigen::RowMajor> proj = getProjections();
  /// ?!

  // Works better without !
  //#pragma omp parallel for
  for (size_t cmp = 0u; cmp < 3u; ++cmp)
  {
    m_rhs.row(cmp) =
      // Momentum
      (m_scene.getReducedMassMatrix() * speed.row(cmp).transpose()
       + m_time_step * (
         // Velocity
         getA().transpose() * proj.row(cmp).transpose()
         // Damping
         - m_damping.asDiagonal() * nextSpeed.row(cmp).transpose()
         // Position
         - getATA() * positions.row(cmp).transpose())).transpose();
  } // cmp
}


/// TODO : clean the damping
/*
SparseMatrix
SimulationSpeed::getCollisionDampingCoefficients() const noexcept
{
  double coefficient = 0;
  if (!m_scene.getFrictionCoefficientTable().empty()  &&
      !m_scene.getFrictionCoefficientTable()[0].empty() )
  {
    coefficient =
      std::atan(0.5 * m_scene.getFrictionCoefficientTable()[0][0]) * 2 /
      M_PI;
  }

  std::size_t vertices_number = m_scene.getNumberOfVertices();
  SparseMatrix result(vertices_number, vertices_number);
  result.setIdentity();
  result -= coefficient * m_scene.getInverseMassMatrix() *
    getCollisionVerticesSelectionMatrix();
  return result;
}
*/



void
SimulationSpeed::exportStats(std::ofstream &of) const noexcept
{
  computeAndWriteStats("LHS computation time (µs)", of,
                       m_lhs_times);
  computeAndWriteStats("LHS update time (µs)", of,
                       m_lhs_update_times);
  computeAndWriteStats("Cholesky computation time (µs)", of,
                       m_cholesky_times);
  SimulationBase::exportStats(of);

}
