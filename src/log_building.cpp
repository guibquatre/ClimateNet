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

#include "log_building.hpp"
#include "StepCallBack.impl.hpp"

#include <algorithm>
#include <iterator>
#include <stdexcept>

StepCallBacks
getLogCallBacksFromJSONDocument(const rapidjson::Document& document)
{
  if (!document.IsObject())
  {
    throw std::invalid_argument("JSON reader - Error : root must be an object");
  }

  auto log_iterator = document.FindMember("log");
  if (log_iterator == document.MemberEnd())
  {
    return StepCallBacks{};
  }

  const rapidjson::Value& log_array = jsonRequireArrayCheck(document, "log");
  return getLogCallBacksFromArray(log_array);
}

StepCallBacks
getLogCallBacksFromArray(const rapidjson::Value& log_array)
{
  std::vector<StepCallBacks::Callback> callbacks;
  std::transform(log_array.Begin(),
                 log_array.End(),
                 std::back_inserter(callbacks),
                 getLogCallBackFromObject);
  StepCallBacks log_callbacks;
  log_callbacks.addCallbacks(callbacks.begin(), callbacks.end());
  return log_callbacks;
}

StepCallBacks::Callback
getLogCallBackFromObject(const rapidjson::Value& log_object)
{
  LogType log_type = getLogTypeFromObject(log_object);
  switch (log_type)
  {
    case LogType::POSITION:
      return getLogPositionCallBackFromObject(log_object);
    case LogType::VELOCITY:
      return getLogVelocityCallBackFromObject(log_object);
    default:
      throw std::invalid_argument("Unsupported log type");
  }
}

LogType
getLogTypeFromObject(const rapidjson::Value& log_object)
{
  std::string log_type_string = jsonRequireString(log_object, "type");

  auto log_type_it = STRING_TO_LOG_TYPE.find(log_type_string);
  if (log_type_it == STRING_TO_LOG_TYPE.end())
  {
    throw std::invalid_argument("Log type \"" + log_type_string
                                + "\" is unsupported");
  }

  return log_type_it->second;
}

StepCallBacks::Callback
getLogPositionCallBackFromObject(const rapidjson::Value& log_object){
  std::vector<size_t> vertices_indices =
    getIndicesFromArray(jsonRequireArrayCheck(log_object, "vertices"));

  return [vertices_indices](size_t, const PhysicScene& scene) {
    std::cout << "Position " << vertices_indices.size() << "\n";
    for (size_t index : vertices_indices)
    {
      std::cout << "v" << index << "\n";
      std::cout << scene.getMesh(0)->getVertexPosition(index) << "\n";
    }
  };
}

StepCallBacks::Callback
getLogVelocityCallBackFromObject(const rapidjson::Value& log_object)
{
  std::vector<size_t> vertices_indices =
    getIndicesFromArray(jsonRequireArrayCheck(log_object, "vertices"));

  return [vertices_indices](size_t, const PhysicScene& scene) {
    std::cout << "Velocity " << vertices_indices.size() << "\n";
    for (size_t index : vertices_indices)
    {
      std::cout << "v" << index << "\n";
      std::cout << scene.getMesh(0)->getVertexSpeed(index) << "\n";
    }
  };
}

std::vector<size_t>
getIndicesFromArray(const rapidjson::Value& array)
{
  std::vector<size_t> indices;
  std::transform(array.Begin(),
                 array.End(),
                 std::back_inserter(indices),
                 std::mem_fn(&rapidjson::Value::GetInt));
  return indices;
}
