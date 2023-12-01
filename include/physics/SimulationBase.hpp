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
#ifndef SIMULATION_BASE_HPP
#define SIMULATION_BASE_HPP

#include <chrono>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <memory>
#include <vector>

#include "physics/AbstractConstraint.hpp"
#include "physics/PhysicScene.hpp"
#include "types.hpp"


/** 
 * @brief Base class for a Projective Dynamics simulation
 */
// TODO: Find a way to use space for the data m_current_step_<something>
// only if there are necessary.
class SimulationBase
{

public:
  /// @brief Function pointer defining the external forces 
  // In a class ? To avoid returning a vector each time
  using ExternalForce = std::function<
  Eigen::Vector3d(const Eigen::Matrix3d&, const Eigen::Vector3d&, double)>;
  /// @brief Filter type for the log ?
  using RelativeVelocitiesFilter =
    std::function<bool(std::size_t, const Eigen::Vector3d&)>;
  /// @brief Typedef for the chrono 
  using IterationDuration = std::chrono::high_resolution_clock::duration;

  /**
   * @brief Constructor
   * @param scene             Set up to simulate
   * @param time_step         Simulation timestep
   * @param iteration_number  Number of iterations 
   * @param external_forces   Function pointer to compute the external forces 
   */
  SimulationBase(PhysicScene& scene,
                 double time_step,
                 std::size_t iteration_number,
                 const ExternalForce& external_force,
                 double air_damping,
                 bool handle_self_collision,
                 double self_collision_tolerance);
  virtual ~SimulationBase() = default;

  /// @brief Core method: performs one simulation step
  void step();

  // IO functions

  /** 
   * @brief Velocity filter for the dumping (copy version)
   * @param filter  Function pointer to the filter
   */
  void setRelativeVelocitiesFilter(const RelativeVelocitiesFilter& filter) noexcept;
  /** 
   * @brief Velocity filter for the dumping (move version)
   * @param filter  Function pointer to the filter
   */
  void setRelativeVelocitiesFilter(RelativeVelocitiesFilter&& filter) noexcept;
  
  /** 
   * @brief Getter on the max relative contact velocity
   * @return Max relative contact velocity
   */
  double getMaxRelativeVelocityNorm() const noexcept;
  /** 
   * @brief Getter on the min relative contact velocity
   * @return Min relative contact velocity
   */
  double getMeanRelativeVelocityNorm() const noexcept;

  /**
   * @brief Getter on the solver erros of each  frame
   * @return Vector concatenating all the solver errors
   */
  const std::vector<double>& getLastStepErrors() const noexcept;
  /**
   * @brief Getter on the duration of each frame 
   * @return Vector concatenating the durations
   */
  const std::vector<IterationDuration>& getLastStepDurations() const noexcept;
  /**
   * @brief Getter on the number of collisions at each frame 
   * @return Vector concatenating the nomber of collisions
   */
  const std::vector<std::size_t>& getLastStepNumberCollisions() const
    noexcept;


  /**
   * @brief Write all the stats in the given ofstream
   */
  virtual void exportStats(std::ofstream &of) const noexcept;
  

protected:

  /// @brief Core of the simulation loop - Initialize the PD step
  virtual void initializeStep() = 0;
  /// @brief Core of the simulation loop - Initialize a PD iteration w/o the collisions
  virtual void buildIteration() = 0;
  /// @brief Core of the simulation loop - Handles the collisions
  virtual void updateCollision() = 0;
  /// @brief Core of the simulation - Solve one iteration of PD
  virtual double solve() = 0;
  /// @brief Core of the simulation - Finalize the timestep and updates the data
  virtual void updateScene() = 0;


  /**
   * @brief Update the current estimation of the position at the next timestep
   */
  virtual void computeCurrentNextPosition() = 0;
  /**
   * @brief Current estimation of the position at the next timestep
   * @return q_{n+1}^k
   */
  const Eigen::VectorXd &getCurrentNextPosition() const noexcept;
  
  /**
   * @brief Update the current estimation of the velocity at the next timestep
   */
  virtual void computeCurrentNextSpeed() = 0;
  /**
   * @brief Current estimation of the velocity at the next timestep
   * @return v_{n+1}^k
   */
  const Eigen::VectorXd& getCurrentNextSpeed() const noexcept;


  /**
   * @brief Compute the current value of the external forces
   */
  void computeCurrentExternalForce() noexcept;
  /**
   * @brief Getter
   * @return The external forces vector
   */
  const Eigen::VectorXd& getCurrentExternalForce() const noexcept;

  /**
   * @brief Compute and return the next speeds of the vertices as if 
   *        there were no constraints
   */
  void computeNextSpeedUnderNoConstraint(
    const Eigen::VectorXd& current_external_forces) noexcept;
  /**
   * @brief Getter
   * @return t_n 
   */
  const Eigen::VectorXd& getNextSpeedUnderNoConstraint() const noexcept;

  const SparseMatrix& getA() const noexcept;

  void computeProjections(const Eigen::VectorXd& positions,
                          const Eigen::VectorXd& speeds);

  const Eigen::Matrix3Xd &getProjections() const noexcept;
  
  /**
   * @brief Computes and store the collisions data
   */
  void baseUpdateColisions() noexcept;


  /// @brief Compute the LHS
  virtual void computeLhs() noexcept = 0;
  /**
   * @brief Getter
   * @return LHS
   */
  const SparseMatrix& getLhs() const noexcept;
  /**
   * @brief Compute the rhs
   * @param
   * @param
   */
  virtual void computeRhs(const Eigen::VectorXd&, const Eigen::VectorXd&)
    noexcept = 0;
  /**
   * @brief Getter
   * @return RHS
   */
  const Eigen::Matrix<double, 3, -1, Eigen::RowMajor>&
    getRhs() const noexcept;
  
  // Update for the IO

  /**
   * @brief Stores the number of collisions for the given iteration
   * @param iteration          ...
   * @param collisions_number  ...
   */
  void updateNumberCollisions(std::size_t iteration,
                              std::size_t collisions_number) noexcept;
  /**
   * @brief Update the private member m_current_step_relative_velocities with 
   *        the relative velocities of the current step. This method should be 
   *        called after the scene velocities have been updated.
   */
  void updateRelativeVelocities() noexcept;
  /**
   * @brief Stores the solver error for the given iteration
   * @param iteration  ...
   * @param error      ...
   */
  void updateErrors(std::size_t iteration, double error) noexcept;
  /**
   * @brief Stores duration for the given iteration
   * @param iteration  ...
   * @param duration   ...
   */
  void updateDurations(
    std::size_t iteration,
    std::chrono::high_resolution_clock::duration duration) noexcept;

  virtual void updateIterationData() noexcept {}


protected:
  // Data

  /// @brief Number of vertices to simulate
  const unsigned int m_nVertices;
  /// @brief Number of degrees of freedom
  const unsigned int m_nDofs;

  /// @brief Simulation time step
  const double m_time_step;
  /// @brief Current simulation time
  double m_current_time;
  /// @brief Current iteration number
  std::size_t m_iteration_number;
	
  /// @brief Simulation scene
  PhysicScene& m_scene;
  /// @brief External force function
  const ExternalForce m_external_force_fun;
  /// @brief ...
  const std::vector<std::shared_ptr<AbstractConstraint>>&
  m_constant_constraints;
	
  /**
   * @brief A predicate that takes in the vertex index, and the vertex 
   *        position and return true if the relative velocity of the vertex 
   *        should be saved.
   */
  RelativeVelocitiesFilter m_relative_velocities_filter;
  /**
   * @brief (IO) Relative velocities of the vertices in collision at the end of 
   *        the last step
   */
  std::vector<Eigen::Vector3d> m_current_step_relative_velocities;
  /// @brief (IO) Number of collisions at each step
  std::vector<std::size_t> m_current_step_number_of_collisions;
  /// @brief (IO) Solver at each step
  std::vector<double> m_current_step_error;
  /// @brief (IO) Duration each step
  std::vector<IterationDuration> m_current_step_duration;

  /// @brief (IO)
  std::vector<double> m_step_times;
  /// @brief (IO)
  std::vector<double> m_rhs_times;
  /// @brief (IO)
  std::vector<double> m_global_times;
  /// @brief (IO)
  std::vector<double> m_iteration_times;
  /// @brief (IO)
  std::vector<double> m_collision_detection_times;
  /// @brief (IO)
  std::vector<size_t> m_collision_numbers;
  /// @brief (IO)
  std::vector<double> m_self_collision_detection_times;
  /// @brief (IO)
  std::vector<size_t> m_self_collision_numbers;
  
  
  //@brief Collision info should be kept sorted in respect to the vertex indices
  //TODO: use a better data structure for sorted collections.
  std::vector<CollisionInfo> m_collisions_infos;


  /// @brief ...
  const bool m_handle_self_collision;
  /// @brief ...
  std::vector<SelfCollisionInfo> m_self_collisions_infos;
  /// @brief ...
  double m_self_collision_tol2;
  
  /// @brief Latest external forces valuation - see getCurrentExternalForce
  Eigen::VectorXd m_current_external_force;
  /// @brief ...
  SparseMatrix m_A;
  /// @brief ...
  Eigen::Matrix3Xd m_projections;

  /// @brief
  double m_damping_coefficient;
  /// @brief
  Eigen::VectorXd m_damping;
  
  SparseMatrix m_contact_matrix_old;
  SparseMatrix m_contact_matrix_new;
  
  /// @brief Predicted explicit velocity with the constant forces only
  Eigen::VectorXd m_t_n;
  /// @brief Current extimate of the next implicit position
  Eigen::VectorXd m_current_next_position;
  /// @brief ...
  Eigen::VectorXd m_current_next_speed;


  // Left for children's equations
  // TODO : make it clean
  /// @brief Right hand side of the global step equation
  Eigen::Matrix<double, 3, -1, Eigen::RowMajor> m_rhs;

  
  /// @brief Left hand side of the global step equation
  SparseMatrix m_lhs;
	
};

#endif
