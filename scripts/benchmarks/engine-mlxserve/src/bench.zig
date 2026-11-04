const std = @import("std");

// Keep these wire models in parity with Rust, C++, and Python (AGENTS.md).
pub const ChatMessage = struct {
    role: []const u8,
    content: ?[]const u8 = null,
    reasoning_content: ?[]const u8 = null,
    tool_calls: ?[]const std.json.Value = null,
    tool_call_id: ?[]const u8 = null,
};

pub const GreedySampling = struct {
    greedy: bool,
};

pub const StochasticSampling = struct {
    top_k: ?i32,
    top_p: ?f32,
    min_p: ?f32,
    temp: ?f32,
    repetition_penalty: ?f32,
    suffix_repetition_length: ?usize,
};

pub const BenchSampling = union(enum) {
    greedy: GreedySampling,
    stochastic: StochasticSampling,

    pub fn jsonParse(a: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !BenchSampling {
        return jsonParseFromValue(a, try std.json.Value.jsonParse(a, source, options), options);
    }

    pub fn jsonParseFromValue(a: std.mem.Allocator, value: std.json.Value, options: std.json.ParseOptions) !BenchSampling {
        if (value == .object and value.object.contains("greedy")) {
            const sampling = try std.json.innerParseFromValue(GreedySampling, a, value, options);
            if (!sampling.greedy) return error.UnexpectedToken;
            return .{ .greedy = sampling };
        }
        return .{ .stochastic = try std.json.innerParseFromValue(StochasticSampling, a, value, options) };
    }
};

pub const ReasoningEffort = enum { disabled, default, low, medium, high, xhigh };

pub const FinishReason = enum { stop, length };

pub const BenchRequest = struct {
    messages: []const ChatMessage,
    tools: ?[]const std.json.Value = null,
    tool_choice: ?std.json.Value = null,
    max_output_tokens: ?usize = null,
    speculative_depth: ?usize = null,
    reasoning_effort: ?ReasoningEffort = null,
};

pub const BenchResponse = struct {
    text: []const u8,
    finish_reason: FinishReason,
    prompt_tokens_count: usize,
    tokens_count: usize,
    time_to_first_token: f64,
    prompt_tps: f64,
    decode_tps: f64,
    tokens_per_forward_pass: ?f64,
    duration: f64,
    energy_gpu: f64,
    memory_phys_footprint: u64,
    memory_resident: u64,
    memory_graphics_total: u64,
};
