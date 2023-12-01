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
#include "scene_building.hpp"

#include <Eigen/Core>

#include <cstdio>
#include <rapidjson/filereadstream.h>

#include "geometry/io.hpp"
#include "jsonHelper.hpp"
#include "physics/MovingMesh.hpp"
#include "physics/PhysicMesh.hpp"
#include "physics/RotatingSphere.hpp"
#include "physics/StaticAxisAlignedBox.hpp"
#include "physics/StaticMesh.hpp"
#include "physics/StaticPlane.hpp"
#include "physics/StaticSphere.hpp"
#include "rapidjson/document.h"

// TODO: Create an exception exclusive to the JSON reader.
// TODO: Centralize the names of the differents field.

PhysicScene
getSceneFromJSONFileName(const char* filename)
{
  PhysicScene scene =
    getSceneFromJSONDocument(getJSONDocumentFromFilename(filename));
  return scene;
}

PhysicScene
getSceneFromJSONDocument(const rapidjson::Document& scene_description)
{
    if (!scene_description.IsObject())
    {
        throw std::invalid_argument("JSON reader - Error : root must be an object");
    }

    const rapidjson::Value& friction_coefficients =
      jsonRequireArrayCheck(scene_description, "Friction coefficients");
    PhysicScene scene(getFrictionCoefficientsFromArray(friction_coefficients));

    const rapidjson::Value& mesh_array = jsonRequireArrayCheck(scene_description, "Meshes");
    for (auto& mesh : getPhysicMeshesFromArray(mesh_array))
    {
        scene.addMesh(std::make_unique<PhysicMesh>(std::move(mesh)));
    }

    const rapidjson::Value& obstacle_array = jsonRequireArrayCheck(scene_description, "Obstacles");
    for (auto& obstacle : getObstaclesFromArray(obstacle_array))
    {
        scene.addObstacle(std::move(obstacle));
    }

    scene.finalize();
    return scene;
}

PhysicScene::FrictionCoefficientTable
getFrictionCoefficientsFromArray(const rapidjson::Value& array)
{
    std::size_t material_number = array.Size();
    PhysicScene::FrictionCoefficientTable coefficients_table(material_number,
                                                             std::vector<double>(material_number));

    for (std::size_t i = 0; i < material_number; ++i)
    {
        if (!array[i].IsArray() || array[i].Size() != material_number)
        {
            throw std::invalid_argument("JSON reader - Error : Friction "
                                        "coefficients must be a square matrix");
        }
        for (std::size_t j = 0; j < material_number; ++j)
        {
            if (!array[i][j].IsDouble())
            {
                throw std::invalid_argument("JSON reader - Error : Friction "
                                            "coefficients must be a square "
                                            "matrix of double");
            }
            coefficients_table[i][j] = array[i][j].GetDouble();
        }
    }
    return coefficients_table;
}

std::vector<PhysicMesh>
getPhysicMeshesFromArray(const rapidjson::Value& array)
{
    std::vector<PhysicMesh> result;
    std::transform(array.Begin(), array.End(), std::back_inserter(result), getPhysicMeshFromObject);
    return result;
}

std::vector<std::unique_ptr<Obstacle>>
getObstaclesFromArray(const rapidjson::Value& array)
{
    std::vector<std::unique_ptr<Obstacle>> result;
    std::transform(array.Begin(), array.End(), std::back_inserter(result), getObstacleFromObject);
    return result;
}

PhysicMesh
getPhysicMeshFromObject(const rapidjson::Value& object)
{
    std::string rest_state_file_name = jsonRequireString(object, "Obj filename");

    std::vector<tinyobj::index_t> triangles;
    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3d> normals;
    std::vector<Eigen::Vector2d> texcoords;
    bool success;

    success = read_obj(rest_state_file_name, vertices, triangles, normals, texcoords);
    if (!success)
    {
        throw std::invalid_argument("Could not load obj file: " + rest_state_file_name);
    }

    PhysicParameters parameter;
    parameter.area_density = jsonRequireDouble(object, "Area Density");
    parameter.bend = jsonRequireDouble(object, "Bending");
    parameter.stretch = jsonRequireDouble(object, "Stretching");

    std::size_t material_identifier = jsonRequireUint(object, "Material Identifier");

    PhysicMesh result(vertices, triangles, parameter, material_identifier);

    auto member_it = object.FindMember("Current state obj filename");
    if (member_it == object.MemberEnd())
    {
        return result;
    }

    std::string current_state_file_name = jsonRequireString(object, "Current state obj filename");
    success = read_obj(current_state_file_name, vertices, triangles, normals, texcoords);
    if (!success)
    {
        throw std::invalid_argument("Could not load obj file: " + current_state_file_name);
    }
    result.setPositions(vertices);

    return result;
}

std::unique_ptr<Obstacle>
getObstacleFromObject(const rapidjson::Value& object)
{
    std::string object_name = jsonRequireString(object, "Type");
    if (object_name == "Sphere")
    {
        return getStaticSphereFromObject(object);
    }
    if (object_name == "Plane")
    {
        return getStaticPlaneFromObject(object);
    }
    if (object_name == "Axis Aligned Box")
    {
        return getStaticAxisAlignedBoxFromObject(object);
    }
    if (object_name == "Mesh")
    {
        return getStaticMeshFromObject(object);
    }
    if (object_name == "Moving Mesh")
    {
        return getMovingMeshFromObject(object);
    }
    if (object_name == "Rotating Sphere")
    {
        return getRotatingSphereFromObject(object);
    }
    throw std::invalid_argument("JSON reader - Error : Invalid type for static object type");
}

std::unique_ptr<Obstacle>
getStaticSphereFromObject(const rapidjson::Value& object)
{
    Eigen::Vector3d position = getVectorFromArray(jsonRequireArrayCheck(object, "Position"));
    double radius = jsonRequireDouble(object, "Radius");
    std::size_t material_identifier = jsonRequireUint(object, "Material Identifier");
    return std::make_unique<StaticSphere>(radius, position, material_identifier);
}

std::unique_ptr<Obstacle>
getStaticAxisAlignedBoxFromObject(const rapidjson::Value& object)
{
    Eigen::Vector3d position = getVectorFromArray(jsonRequireArrayCheck(object, "Position"));
    std::size_t material_identifier = jsonRequireUint(object, "Material Identifier");
    Eigen::Vector3d size = getVectorFromArray(jsonRequireArrayCheck(object, "Size"));
    return std::make_unique<StaticAxisAlignedBox>(position, size, material_identifier);
}

std::unique_ptr<Obstacle>
getStaticPlaneFromObject(const rapidjson::Value& object)
{
    Eigen::Vector3d point = getVectorFromArray(jsonRequireArrayCheck(object, "Point"));
    Eigen::Vector3d normal = getVectorFromArray(jsonRequireArrayCheck(object, "Normal"));
    std::size_t material_identifier = jsonRequireUint(object, "Material Identifier");
    return std::make_unique<StaticPlane>(normal, point, material_identifier);
}

std::unique_ptr<Obstacle>
getStaticMeshFromObject(const rapidjson::Value& object)
{
    // TODO: Factorize with getPhysicMeshFromObject
    std::string filepath = jsonRequireString(object, "Path");

    std::vector<tinyobj::index_t> triangles;
    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3d> normals;
    std::vector<Eigen::Vector2d> texcoords;

    const bool success = read_obj(filepath, vertices, triangles, normals, texcoords);
    if (!success)
    {
        throw std::invalid_argument("Could not load obj file: " + filepath);
    }

    std::shared_ptr<Mesh> obstaclePtr = std::make_shared<Mesh>(vertices, triangles);

    const double tol = jsonRequireDouble(object, "Threshold");

    const std::size_t material_identifier = jsonRequireUint(object, "Material Identifier");

    return std::make_unique<StaticMesh>(obstaclePtr, tol, material_identifier);
}

std::unique_ptr<Obstacle>
getMovingMeshFromObject(const rapidjson::Value& object)
{
    const std::string base_name = jsonRequireString(object, "Obj Prefix");
    const std::size_t suffix_size = jsonRequireUint(object, "Suffix Size");
    const std::size_t number_frame = jsonRequireUint(object, "Number Frame");
    const std::size_t material_identifier = jsonRequireUint(object, "Material Identifier");
    const double time_between_frame = jsonRequireDouble(object, "Time Between Frame");
    const double collision_tolerance = jsonRequireDouble(object, "Collision tolerance");
    std::stringstream ss;

    std::vector<std::string> main_frames_filenames;
    std::vector<double> main_frames_times;
    for (std::size_t frame_index = 0; frame_index < number_frame; ++frame_index)
    {
        ss << base_name;
        if (suffix_size > 0)
        {
            ss << std::setfill('0') << std::setw(suffix_size);
        }
        ss << frame_index << ".obj";
        main_frames_filenames.push_back(ss.str());
        main_frames_times.push_back(frame_index * time_between_frame);

        ss.str("");
    }

    return std::make_unique<MovingMesh>(
      main_frames_filenames, main_frames_times, collision_tolerance, material_identifier);
}

std::unique_ptr<Obstacle>
getRotatingSphereFromObject(const rapidjson::Value& object)
{
    double radius = jsonRequireDouble(object, "Radius");
    Eigen::Vector3d center = getVectorFromArray(jsonRequireArrayCheck(object, "Position"));
    Eigen::Vector3d rotation_axe =
      getVectorFromArray(jsonRequireArrayCheck(object, "Rotation Axe"));
    double radial_speed = jsonRequireDouble(object, "Radial Speed");
    double rotation_start_time = jsonRequireDouble(object, "Rotation Start Time");
    std::size_t material_identifier = jsonRequireUint(object, "Material Identifier");

    return std::make_unique<RotatingSphere>(Sphere(radius, std::move(center)),
                                            rotation_axe,
                                            radial_speed,
                                            rotation_start_time,
                                            material_identifier);
}

Eigen::Vector3d
getVectorFromArray(const rapidjson::Value& array)
{
    if (array.Size() != 3u)
    {
        throw std::invalid_argument(
          "JSON reader - Error : Array representing vector must be of size 3");
    }

    Eigen::Vector3d result;
    for (std::size_t i = 0; i < 3u; ++i)
    {
        if (!array[i].IsDouble())
        {
            throw std::invalid_argument("JSON reader - Error : Array "
                                        "representing vector must contain "
                                        "doubles");
        }
        result[i] = array[i].GetDouble();
    }
    return result;
}
