#ifndef LOG_BUILDING_HPP
#define LOG_BUILDING_HPP

#include "StepCallBack.hpp"
#include "jsonHelper.hpp"
#include "rapidjson/document.h"

#include <unordered_map>

enum class LogType
{
  VELOCITY,
  POSITION
};

const static std::unordered_map<std::string, LogType> STRING_TO_LOG_TYPE = {
  { "velocity", LogType::VELOCITY },
  { "position", LogType::POSITION }
};

StepCallBacks
getLogCallBacksFromJSONDocument(const rapidjson::Document& document);

StepCallBacks
getLogCallBacksFromArray(const rapidjson::Value& log_array);

StepCallBacks::Callback
getLogCallBackFromObject(const rapidjson::Value& log_object);

LogType
getLogTypeFromObject(const rapidjson::Value& log_object);

StepCallBacks::Callback
getLogPositionCallBackFromObject(const rapidjson::Value& log_object);

StepCallBacks::Callback
getLogVelocityCallBackFromObject(const rapidjson::Value& log_object);

std::vector<size_t>
getIndicesFromArray(const rapidjson::Value& array);

#endif
