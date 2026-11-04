#ifndef __benchmarks_common_hpp__
#define __benchmarks_common_hpp__

#include <cstdio>
#include <glaze/glaze.hpp>
#include <stdexcept>
#include <string>

#include "bench.hpp"
#include "memory_counters.h"

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    virtual BenchResponse execute(const BenchRequest& request) const = 0;
};

std::FILE* redirect_stdout_to_stderr();

memory_counters_t collect_memory_counters();
double collect_gpu_energy();
void validate_sampling(const BenchSampling& sampling);

template <typename ModelConfig>
ModelConfig read_model_config(const std::string& path) {
    ModelConfig config;
    std::string buffer;
    if (const auto error = glz::read_file_json<glz::opts{.error_on_missing_keys = true}>(config, path, buffer)) {
        throw std::runtime_error(path + ": " + glz::format_error(error, buffer));
    }
    validate_sampling(config.sampling);
    return config;
}

void run_loop(
    const InferenceEngine& engine,
    std::FILE* output
);

#endif  // __benchmarks_common_hpp__
