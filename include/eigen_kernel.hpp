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
#ifndef EIGEN_KERNEL_HPP
#define EIGEN_KERNEL_HPP

/** @file
 * @brief Defines a CGAL cartesian kernel where the points are Eigen vectors.
 * Most CGAL libraries are parametrized by a Kernel which describe the space in
 * which the computation takes place. The file provide a Kernel EigenKernel in
 * which a point in space is described by an Eigen::Vector3D. This Kernel allows
 * us to use the convenient CGAL algorithms on our Mesh.
 */

#include <CGAL/Bbox_3.h>
#include <CGAL/Cartesian.h>
#include <CGAL/Kernel/Return_base_tag.h>
#include <CGAL/Kernel/Type_equality_wrapper.h>
#include <CGAL/Origin.h>
#include <Eigen/Core>

template<class Construct_bbox_3>
class EigenConstruct_bbox_3 : public Construct_bbox_3
{
public:
    using Construct_bbox_3::operator();

    CGAL::Bbox_3 operator()(const Eigen::Vector3d& point) const
    {
        return CGAL::Bbox_3(
          point.x(), point.y(), point.z(), point.x(), point.y(), point.z());
    }
};

class EigenConstruct_coord_iterator
{
public:
    const Eigen::Vector3d::Scalar* operator()(const Eigen::Vector3d& point)
    {
        return point.data();
    }

    const Eigen::Vector3d::Scalar* operator()(const Eigen::Vector3d& point, int)
    {
        return point.data() + point.size();
    }
};

template<typename Kernel, typename OldKernel>
class EigenConstruct_point_3
{
    using Point_3 = typename Kernel::Point_3;
    using Rep = typename Point_3::Rep;
    using RT = typename Kernel::RT;
    using Line_3 = typename Kernel::Line_3;

public:
    using result_type = Point_3;

    Rep operator()(CGAL::Return_base_tag, CGAL::Origin o) const
    {
        return Rep(o);
    }

    Rep operator()(CGAL::Return_base_tag,
                   const RT& x,
                   const RT& y,
                   const RT& z) const
    {
        return Rep(x, y, z);
    }

    Rep operator()(CGAL::Return_base_tag,
                   const RT& x,
                   const RT& y,
                   const RT& z,
                   const RT& w) const
    {
        return Rep(x, y, z, w);
    }

    Point_3 operator()(const CGAL::Origin&) const
    {
        return Eigen::Vector3d(0, 0, 0);
    }

    Point_3 operator()(const RT& x, const RT& y, const RT& z) const
    {
        return Eigen::Vector3d(x, y, z);
    }

    const Point_3& operator()(const Point_3& p) const { return p; }

    template<typename OtherDerived>
    Point_3 operator()(const Eigen::EigenBase<OtherDerived>& other) const
    {
        return Eigen::Vector3d(other);
    }

    Point_3 operator()(const Line_3& l) const
    {
        typename OldKernel::Construct_point_3 base_operator;
        return base_operator(l);
    }

    Point_3 operator()(const Line_3& l, int i) const
    {
        typename OldKernel::Construct_point_3 base_operator;
        return base_operator(l, i);
    }
};

template<typename Kernel, typename Kernel_Base>
class EigenCartesian_base : public Kernel_Base::template Base<Kernel>::Type
{
    using OldKernel = typename Kernel_Base::template Base<Kernel>::Type;

public:
    using Point_3 = Eigen::Vector3d;
    using Construct_point_3 = EigenConstruct_point_3<Kernel, OldKernel>;
    using Cartesian_const_iterator_3 = const Eigen::Vector3d::Scalar*;
    using Construct_cartesian_const_iterator_3 = EigenConstruct_coord_iterator;
    using Construct_bbox_3 =
      EigenConstruct_bbox_3<typename OldKernel::Construct_bbox_3>;

    Construct_point_3 construct_point_3_object() const
    {
        return Construct_point_3();
    }

    Construct_bbox_3 construct_bbox_3_object() const
    {
        return Construct_bbox_3();
    }

    Construct_cartesian_const_iterator_3 construct_cartesian_const_iterator_3_object()
      const
    {
        return Construct_cartesian_const_iterator_3();
    }

    template<typename Kernel2>
    struct Base
    {
        using Type = EigenCartesian_base<Kernel2, Kernel_Base>;
    };
};

/**
 * CGAL Cartesian Kernel where the points are Eigen Vectors.
 */
struct EigenKernel
  : public CGAL::Type_equality_wrapper<
      EigenCartesian_base<EigenKernel,
                          CGAL::Cartesian<Eigen::Vector3d::Scalar>>,
      EigenKernel>
{
};

#endif
