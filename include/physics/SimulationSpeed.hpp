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
#ifndef SIMULATION_SPEED_HPP
#define SIMULATION_SPEED_HPP

#include<Eigen/SparseCholesky>
#include "physics/SimulationBase.hpp"


/**
 * @brief Class implementing the reformulation in speed 
 *        of the projective dynamics
 */

class SimulationSpeed : public SimulationBase
{
public:
  
  /**
   * @brief Constructor
   * @param scene             Set up to simulate
   * @param time_step         Simulation timestep
   * @param iteration_number  Number of iterations 
   * @param external_forces   Function pointer to compute the external forces 
   */
  SimulationSpeed(PhysicScene& scene,
                  double time_step,
                  std::size_t iteration_number,
                  const ExternalForce& external_force,
                  double air_damping,
                  bool handle_self_collision,
                  double self_collision_tolerance);


  /**
   * @brief Write all the stats in the given ofstream
   */
  virtual void exportStats(std::ofstream &of) const noexcept override;
  
protected:
  /// @brief Initialize a time-step
  virtual void initializeStep() noexcept override;
  /// @brief Perform the local steps and build the global equation w/o contact
  virtual void buildIteration() noexcept override;
  /// @brief Update the collisions terms in the global equation
  virtual void updateCollision() noexcept override;
  /// @brief Solve the global equation
  virtual double solve() noexcept override;
  /// @brief Finalize a time-step
  virtual void updateScene() noexcept override;
  
  /// @brief 
  virtual void computeCurrentNextPosition() noexcept override;

  /// @brief 
  virtual void computeCurrentNextSpeed() noexcept override;
  
  const SparseMatrix& getATA() const noexcept;
  
  /// @brief Compute the lhs of the global equation w/o contact
  virtual void computeLhs() noexcept override;
  
  /**
   * @brief Compute the rhs of the global equation w/o contact
   * @param current_next_speed
   * @param next_speed_under_no_constraint
   */
  virtual void computeRhs(
    const Eigen::VectorXd& current_next_speed,
    const Eigen::VectorXd& next_speed_under_no_constraint) noexcept override;
  
  /**
   * @brief Viscous damping at the contact points
   * @return Damping matrix
   */
  //SparseMatrix getCollisionDampingCoefficients() const noexcept;

  /// @brief Set of collision informations
  std::vector<std::unique_ptr<AbstractConstraint>> m_collision_constraints;

  /// @brief A^T A
  SparseMatrix m_ATA;

  /// @brief Linear solver for the global step
  Eigen::SimplicialLLT<SparseMatrix> m_solver;
protected:
  /// @brief (IO)
  std::vector<double> m_lhs_times;
  /// @brief (IO)
  std::vector<double> m_lhs_update_times;
  /// @brief (IO)
  std::vector<double> m_cholesky_times;
};

#endif // SIMULATION_SPEED_HPP
