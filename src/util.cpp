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
#include "util.hpp"

#include <Eigen/Core>

Eigen::Block<Eigen::MatrixXd, 3, 3>
getMatrix3dBlock(Eigen::MatrixXd& matrix, int row, int column)
{
    return matrix.block<3, 3>(3 * row, 3 * column);
}

const Eigen::Block<const Eigen::MatrixXd, 3, 3>
getMatrix3dBlock(const Eigen::MatrixXd& matrix, int row, int column)
{
    return matrix.block<3, 3>(3 * row, 3 * column);
}

Eigen::VectorBlock<Eigen::VectorXd, 3>
getVector3dBlock(Eigen::VectorXd& vector, int index)
{
    return vector.segment<3>(3 * index);
}

const Eigen::VectorBlock<const Eigen::VectorXd, 3>
getVector3dBlock(const Eigen::VectorXd& vector, int index)
{
    return vector.segment<3>(3 * index);
}

void
resetSparseMatrix(SparseMatrix& matrix)
{
    matrix *= 0;
}

/**
 * Return the number of 3d vector stored in the given vector.
 */
std::size_t
getVector3dSize(const Eigen::VectorXd& vector)
{
    return vector.size() / 3;
}

Eigen::MatrixXd
getMatrixOn3dVector(std::size_t nb_vector)
{
    return Eigen::MatrixXd(nb_vector * 3, nb_vector * 3);
}
