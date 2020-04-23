#include <iostream>

#include "korali.hpp"
#include "model/model.hpp"

int main(int argc, char **argv)
{
  auto k = korali::Engine();
  auto e = korali::Experiment();

  std::cout << "Engine and Experiment loaded" << std::endl;
  std::cout << "fitness(3.14) = " << fitness(3.14) << std::endl;

  //k.run(e);
  return 0;
}
