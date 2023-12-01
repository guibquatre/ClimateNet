#ifndef STEP_CALLBACKS_IMPL_HPP
#define STEP_CALLBACKS_IMPL_HPP

#include "StepCallBack.hpp"

template<typename SinglePassIterator>
void
StepCallBacks::addCallbacks(SinglePassIterator begin, SinglePassIterator end)
{
  while (begin != end)
  {
    addCallback(*begin);
    ++begin;
  }
}

#endif
