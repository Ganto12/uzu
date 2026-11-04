#ifndef __benchmarks_bench_hpp__
#define __benchmarks_bench_hpp__

#include <array>
#include <cstddef>
#include <cstdint>
#include <glaze/json/generic.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

struct ChatMessage {
    std::string role;
    std::optional<std::string> content = std::nullopt;
    std::optional<std::string> reasoning_content = std::nullopt;
    std::optional<std::vector<glz::generic_u64>> tool_calls = std::nullopt;
    std::optional<std::string> tool_call_id = std::nullopt;
};

struct GreedySampling {
    bool greedy;
};

struct StochasticSampling {
    std::optional<int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> temp;
    std::optional<float> repetition_penalty;
    std::optional<size_t> suffix_repetition_length;
};

template <>
struct glz::meta<StochasticSampling> {
    static constexpr bool requires_key(
        std::string_view,
        bool
    ) {
        return true;
    }
};

using BenchSampling = std::variant<GreedySampling, StochasticSampling>;

enum class ReasoningEffort { disabled, default_effort, low, medium, high, xhigh };

template <>
struct glz::meta<ReasoningEffort> {
    using enum ReasoningEffort;
    static constexpr std::array keys{"disabled", "default", "low", "medium", "high", "xhigh"};
    static constexpr std::array value{disabled, default_effort, low, medium, high, xhigh};
};

enum class FinishReason { stop, length };

template <>
struct glz::meta<FinishReason> {
    using enum FinishReason;
    static constexpr auto value = glz::enumerate(stop, length);
};

struct BenchRequest {
    std::vector<ChatMessage> messages;
    std::optional<std::vector<glz::generic_u64>> tools = std::nullopt;
    std::optional<glz::generic_u64> tool_choice = std::nullopt;

    std::optional<size_t> max_output_tokens = std::nullopt;
    std::optional<size_t> speculative_depth = std::nullopt;
    std::optional<ReasoningEffort> reasoning_effort = std::nullopt;
};

struct BenchResponse {
    std::string text;
    FinishReason finish_reason;
    size_t prompt_tokens_count;
    size_t tokens_count;
    double time_to_first_token;
    double prompt_tps;
    double decode_tps;
    std::optional<double> tokens_per_forward_pass;
    double duration;
    double energy_gpu;
    uint64_t memory_phys_footprint;
    uint64_t memory_resident;
    uint64_t memory_graphics_total;
};

#endif  // __benchmarks_bench_hpp__
