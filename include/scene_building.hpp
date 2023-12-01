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
#ifndef SCENE_BUILDING_HPP
#define SCENE_BUILDING_HPP

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

#include "physics/AttachmentConstraint.hpp"
#include "physics/PhysicScene.hpp"

/** @file
 * Declare utilities to build a scene from a JSON configuration file. When passed
 * an ill-formed JSON configuration, the functions will throw an
 * `std::invalid_argument`. For documentation on well-formed JSON configuration
 * see the `Configuration.md` file in the repository.
 */

PhysicScene
getSceneFromJSONFileName(const char* filename);
PhysicScene
getSceneFromJSONDocument(const rapidjson::Document& document);

PhysicScene::FrictionCoefficientTable
getFrictionCoefficientsFromArray(const rapidjson::Value& object);

std::vector<PhysicMesh>
getPhysicMeshesFromArray(const rapidjson::Value& array);
std::vector<std::unique_ptr<Obstacle>>
getObstaclesFromArray(const rapidjson::Value& array);
std::vector<AttachmentConstraint>
getUserConstraintFromArray(const rapidjson::Value& array);

PhysicMesh
getPhysicMeshFromObject(const rapidjson::Value& object);

std::unique_ptr<Obstacle>
getObstacleFromObject(const rapidjson::Value& object);
std::unique_ptr<Obstacle>
getStaticSphereFromObject(const rapidjson::Value& object);
std::unique_ptr<Obstacle>
getStaticAxisAlignedBoxFromObject(const rapidjson::Value& object);
std::unique_ptr<Obstacle>
getStaticPlaneFromObject(const rapidjson::Value& object);
std::unique_ptr<Obstacle>
getStaticMeshFromObject(const rapidjson::Value& object);
std::unique_ptr<Obstacle>
getMovingMeshFromObject(const rapidjson::Value& object);
std::unique_ptr<Obstacle>
getRotatingSphereFromObject(const rapidjson::Value& object);

Eigen::Vector3d
getVectorFromArray(const rapidjson::Value& array);

#endif
