#ifndef STEP_CALLBACKS_HPP
#define STEP_CALLBACKS_HPP

#include "physics/PhysicScene.hpp"

class StepCallBacks
{
public:
  using Callback = std::function<void(size_t, const PhysicScene&)>;

  StepCallBacks() = default;
  StepCallBacks(const StepCallBacks&) = default;
  StepCallBacks(StepCallBacks&&) = default;
  StepCallBacks& operator=(const StepCallBacks&) = default;
  StepCallBacks& operator=(StepCallBacks&&) = default;

  void addCallback(const Callback& callback);
  void addCallback(Callback&& callback);
  template<typename SinglePassIterator>
  void addCallbacks(SinglePassIterator begin, SinglePassIterator end);
  void operator()(size_t step, const PhysicScene& scene) const;

private:
  std::vector<Callback> m_callbacks;
};

#endif
