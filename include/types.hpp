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
#ifndef TYPES_HPP
#define TYPES_HPP

#include <Eigen/Core>
#include <Eigen/Sparse>

/** @file
 * Defines the types used throughout the code base for handling matrices. 
 */

/// @brief Typedef for sparse matrices
using SparseMatrix = Eigen::SparseMatrix<double>;

/// @brief Helper
using SparseTriplet = Eigen::Triplet<double>;

/// @brief Typedef for dense matrices
using Matrix = Eigen::MatrixXd;
// TODO: inconsistant with the vectors


#endif