#include "StepCallBack.hpp"

void
StepCallBacks::addCallback(const Callback& callback)
{
  m_callbacks.push_back(callback);
}

void
StepCallBacks::addCallback(Callback&& callback)
{
  m_callbacks.push_back(std::move(callback));
}

void
StepCallBacks::operator()(size_t step, const PhysicScene& scene) const
{
  for (const auto& callback : m_callbacks)
  {
    callback(step, scene);
  }
}
