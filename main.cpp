#include "MetalEngine.hpp"
#include <filesystem>

int main(int argc, char *argv[]) {
  std::cout << "Current path is " << std::filesystem::current_path() << '\n';
  MTLEngine engine;
  engine.init();
  engine.run();
  engine.cleanup();
}
