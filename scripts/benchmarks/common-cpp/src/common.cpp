#include "common.hpp"

#include <unistd.h>

#include <glaze/glaze.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

#include "apple_temp_sensors.h"
#include "energy_counters.h"

std::FILE* redirect_stdout_to_stderr() {
    std::cout.flush();
    std::fflush(stdout);

    // Keep the original stdout for responses; send C and C++ library output to stderr.
    const int output_fd = dup(STDOUT_FILENO);
    if (output_fd < 0) {
        throw std::runtime_error("Failed to duplicate stdout");
    }

    std::FILE* output = fdopen(output_fd, "w");
    if (!output) {
        close(output_fd);
        throw std::runtime_error("Failed to open response output");
    }

    if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
        std::fclose(output);
        throw std::runtime_error("Failed to redirect engine output to stderr");
    }
    return output;
}

memory_counters_t collect_memory_counters() {
    memory_counters_t memory_counters{};
    const kern_return_t result = get_memory_counters(&memory_counters, false);
    if (result != KERN_SUCCESS) {
        throw std::runtime_error(
            std::string{"Failed to collect memory counters: "} + memory_counters_error_string(result)
        );
    }
    return memory_counters;
}

double collect_gpu_energy() {
    double joules = 0;
    const kern_return_t result = get_gpu_energy(&joules);
    if (result != KERN_SUCCESS) {
        throw std::runtime_error("Failed to collect GPU energy (" + std::to_string(result) + ")");
    }
    return joules;
}

static BenchRequest get_request_from_json(const std::string& json_text) {
    BenchRequest request;
    if (const auto error =
            glz::read<glz::opts_validate{{glz::opts{.error_on_missing_keys = true}}}>(request, json_text)) {
        throw std::runtime_error(glz::format_error(error, json_text));
    }
    return request;
}

void validate_sampling(const BenchSampling& sampling) {
    if (const auto* greedy = std::get_if<GreedySampling>(&sampling)) {
        if (!greedy->greedy)
            throw std::runtime_error("sampling.greedy must be true");
    } else {
        const auto& stochastic = std::get<StochasticSampling>(sampling);
        if (stochastic.temp == 0) {
            throw std::runtime_error("sampling.temp cannot be 0; use sampling.greedy: true");
        }
        if (stochastic.repetition_penalty && !stochastic.suffix_repetition_length) {
            throw std::runtime_error("repetition_penalty requires suffix_repetition_length");
        }
    }
}

static void validate_request(const BenchRequest& request) {
    if (request.messages.empty()) {
        throw std::runtime_error("Input chat is empty");
    }
}

void run_loop(
    const InferenceEngine& engine,
    std::FILE* output
) {
    std::string line;
    std::cin.clear();
    while (std::getline(std::cin, line)) {
        if (line.find_first_not_of(" \t\r\n\v\f") == std::string::npos) {
            continue;
        }

        const kern_return_t cooldown_result = await_cooldown();
        if (cooldown_result != KERN_SUCCESS) {
            throw std::runtime_error("Cooldown failed (" + std::to_string(cooldown_result) + ")");
        }

        std::string response_json;
        try {
            const auto request = get_request_from_json(line);
            validate_request(request);
            const auto response = engine.execute(request);
            if (const auto error = glz::write<glz::opts{.skip_null_members = false}>(response, response_json)) {
                throw std::runtime_error("Failed to serialize output: " + glz::format_error(error));
            }
        } catch (const std::exception& error) {
            std::cerr << "Failed to process request: " << error.what() << std::endl;
            continue;
        }

        if (std::fwrite(response_json.data(), 1, response_json.size(), output) != response_json.size() ||
            std::fputc('\n', output) == EOF || std::fflush(output) != 0) {
            throw std::runtime_error("Failed to write response to stdout");
        }
    }

    if (std::cin.bad()) {
        throw std::runtime_error("Failed to read request from stdin");
    }
}
