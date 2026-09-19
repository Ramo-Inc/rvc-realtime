// Experimental single-threaded ABI. No Python headers or Python runtime.
#include <torch/script.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/ir/constants.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/ops/_weight_norm.h>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <unordered_map>
#define NOMINMAX
#include <windows.h>
#include <psapi.h>

namespace {
thread_local std::string last_error;
struct Probe {
    torch::jit::Module module;
    std::unordered_map<std::string, at::Tensor> weights;
    bool ready = false;
    bool graph_enabled = false;
    bool poisoned = false;
    std::optional<c10::cuda::CUDAStream> graph_stream;
    size_t graph_output_len = 0;
    std::vector<torch::jit::IValue> static_args;
    at::Tensor static_output;
    std::unique_ptr<at::cuda::CUDAGraph> cuda_graph;
    explicit Probe(const char* path) : module(torch::jit::load(path, at::Device(at::kCUDA, 0))) {}
};
void replace_scalar_assignment(std::shared_ptr<torch::jit::Graph> graph) {
    torch::jit::Node* copy = nullptr;
    for (auto* node : graph->nodes()) {
        if (node->kind() == c10::Symbol::fromQualString("aten::copy_")) {
            TORCH_CHECK(!copy, "multiple copy nodes; unrecognized generator");
            copy = node;
        }
    }
    TORCH_CHECK(copy && copy->inputs().size() == 3, "missing scalar assignment");
    auto* dest = copy->input(0);
    auto* source = copy->input(1)->node();
    TORCH_CHECK(dest->node()->kind() == c10::Symbol::fromQualString("aten::select") &&
        source->kind() == c10::Symbol::fromQualString("aten::tensor"), "unexpected assignment");
    auto* select = dest->node();
    auto constant = [](torch::jit::Value* v) { return torch::jit::toIValue(v).value(); };
    TORCH_CHECK(select->input(0)->node()->kind() == c10::Symbol::fromQualString("aten::rand") &&
        constant(select->input(1)).toInt() == -1 && constant(select->input(2)).toInt() == 0 &&
        constant(source->input(0)).toInt() == 0 && !constant(copy->input(2)).toBool(),
        "unexpected scalar assignment operands");
    TORCH_CHECK(source->inputs().size() == 4 &&
        source->input(1)->node()->kind() == c10::Symbol::fromQualString("prim::dtype") &&
        source->input(2)->node()->kind() == c10::Symbol::fromQualString("prim::device") &&
        source->input(1)->node()->input(0) == dest && source->input(2)->node()->input(0) == dest &&
        !constant(source->input(3)).toBool(), "scalar dtype/device differs from destination");
    auto* replacement = graph->create(c10::Symbol::fromQualString("aten::fill_"), {dest, source->input(0)}, 1);
    replacement->output()->setType(copy->output()->type());
    replacement->insertBefore(copy);
    copy->output()->replaceAllUsesWith(replacement->output());
    copy->destroy();
    torch::jit::EliminateDeadCode(graph);
    graph->lint();
}
at::ScalarType dtype(int value) {
    switch (value) {
        case 0: return at::kHalf;
        case 1: return at::kFloat;
        case 2: return at::kLong;
        case 3: return at::kByte;
        default: throw std::runtime_error("unsupported dtype");
    }
}
at::Tensor tensor(const void* data, const int64_t* dims, size_t rank, int type, bool gpu, bool hot_input = false) {
    auto value = at::from_blob(const_cast<void*>(data), at::IntArrayRef(dims, rank), at::TensorOptions().dtype(dtype(type)));
    // CUDA transfer is blocking (non_blocking=false): the caller's borrowed
    // bytes remain alive through the copy. Avoid a redundant parallel CPU clone
    // on every features block. CPU tensors still need owned storage.
    if (gpu && hot_input) return value.to(at::Device(at::kCUDA, 0), dtype(type), false, true);
    value = value.clone();
    return gpu ? value.to(at::Device(at::kCUDA, 0)) : value;
}
template<class Fn> int checked(Fn fn) {
    try { at::NoGradGuard no_grad; fn(); last_error.clear(); return 0; }
    catch (const std::exception& e) { last_error = e.what(); return -1; }
    catch (...) { last_error = "unknown native exception"; return -1; }
}
}

extern "C" {
const char* rvc_error() { return last_error.c_str(); }
const char* rvc_modules() {
    static thread_local std::string paths;
    paths.clear();
    HMODULE modules[4096];
    DWORD needed = 0;
    if (!K32EnumProcessModules(GetCurrentProcess(), modules, sizeof(modules), &needed) || needed > sizeof(modules)) return nullptr;
    for (size_t i = 0; i < needed / sizeof(HMODULE); ++i) {
        wchar_t name[32768];
        DWORD len = GetModuleFileNameW(modules[i], name, 32768);
        if (!len || len == 32768) return nullptr;
        int count = WideCharToMultiByte(CP_UTF8, 0, name, len, nullptr, 0, nullptr, nullptr);
        std::string utf8(count, '\0');
        WideCharToMultiByte(CP_UTF8, 0, name, len, utf8.data(), count, nullptr, nullptr);
        paths += utf8 + "\n";
    }
    return paths.c_str();
}
int rvc_create(const char* path, void** out) {
    return checked([&] {
        auto p = std::make_unique<Probe>(path);
        const char* option = std::getenv("RVC_NATIVE_CUDA_GRAPH");
        p->graph_enabled = option && std::strcmp(option, "1") == 0;
        *out = p.release();
    });
}
int rvc_set_graph(void* ptr, bool enabled) {
    return checked([&] {
        auto& p = *static_cast<Probe*>(ptr);
        TORCH_CHECK(!p.ready, "already optimized");
        p.graph_enabled = enabled;
    });
}
void rvc_destroy(void* ptr) { delete static_cast<Probe*>(ptr); }
int rvc_weight(void* ptr, const char* name, const void* data, const int64_t* dims, size_t rank, int type) {
    return checked([&] {
        auto& p = *static_cast<Probe*>(ptr);
        TORCH_CHECK(!p.ready, "already optimized");
        TORCH_CHECK(p.weights.emplace(name, tensor(data, dims, rank, type, true).to(at::kFloat)).second, "duplicate weight");
    });
}
int rvc_finalize(void* ptr) {
    return checked([&] {
        auto& p = *static_cast<Probe*>(ptr);
        TORCH_CHECK(!p.ready, "already optimized");
        for (auto param : p.module.named_parameters()) {
            // Pinned models.py:728 infer never calls enc_q; only ignored
            // training forward uses it. Distributed voices omit these weights.
            if (param.name.rfind("enc_q.", 0) == 0) continue;
            auto found = p.weights.find(param.name);
            at::Tensor value;
            if (found != p.weights.end()) {
                value = found->second;
            } else {
                auto g = p.weights.find(param.name + "_g");
                auto v = p.weights.find(param.name + "_v");
                TORCH_CHECK(g != p.weights.end() && v != p.weights.end(), "missing weight ", param.name);
                value = at::_weight_norm(v->second, g->second, 0);
            }
            if (param.name == "emb_g.weight" && param.value.size(0) == 1) {
                TORCH_CHECK(value.dim() == 2 && value.size(0) >= 1, "invalid speaker embedding");
                value = value.narrow(0, 0, 1);
            }
            TORCH_CHECK(value.sizes() == param.value.sizes(), "weight shape mismatch: ", param.name);
            param.value.copy_(value);
        }
        p.weights.clear();
        p.module.eval();
        // Match Python _freeze.py. C++ freeze() unconditionally accesses
        // forward, but this original RVC module exports only infer.
        p.module = torch::jit::freeze_module(p.module, {"infer"});
        auto graph = p.module.get_method("infer").graph();
        torch::jit::OptimizeFrozenGraph(graph, true);
        p.module = torch::jit::optimize_for_inference(p.module, {"infer"});
        if (p.graph_enabled) replace_scalar_assignment(p.module.get_method("infer").graph());
        p.ready = true;
    });
}
int rvc_rng(const uint8_t* cpu, size_t cpu_len, const uint8_t* cuda, size_t cuda_len) {
    return checked([&] {
        int64_t n = static_cast<int64_t>(cpu_len);
        auto cg = at::detail::getDefaultCPUGenerator();
        cg.set_state(tensor(cpu, &n, 1, 3, false));
        n = static_cast<int64_t>(cuda_len);
        auto gg = at::cuda::detail::getDefaultCUDAGenerator(0);
        gg.set_state(tensor(cuda, &n, 1, 3, false));
    });
}
int rvc_seed(uint64_t seed) {
    return checked([&] {
        auto cpu = at::detail::getDefaultCPUGenerator();
        auto gpu = at::cuda::detail::getDefaultCUDAGenerator(0);
        { std::lock_guard<std::mutex> lock(cpu.mutex()); cpu.set_current_seed(seed); }
        { std::lock_guard<std::mutex> lock(gpu.mutex()); gpu.set_current_seed(seed); }
    });
}
int rvc_rng_matches(const uint8_t* cpu, size_t cpu_len, const uint8_t* cuda, size_t cuda_len) {
    return checked([&] {
        auto a = at::detail::getDefaultCPUGenerator().get_state();
        auto b = at::cuda::detail::getDefaultCUDAGenerator(0).get_state();
        TORCH_CHECK(a.nbytes() == cpu_len && b.nbytes() == cuda_len, "RNG state length mismatch");
        TORCH_CHECK(std::memcmp(a.data_ptr(), cpu, cpu_len) == 0 && std::memcmp(b.data_ptr(), cuda, cuda_len) == 0, "RNG state mismatch");
    });
}
int rvc_infer(void* ptr, const void* features, int64_t frames, const void* pitch,
              const void* pitchf, int64_t skip, int64_t ret, int64_t formant,
              float* output, size_t output_len) {
    int code = checked([&] {
        auto& p = *static_cast<Probe*>(ptr);
        TORCH_CHECK(p.ready, "not initialized");
        TORCH_CHECK(!p.poisoned, "generator failed; recreate it");
        std::optional<c10::cuda::CUDAStreamGuard> stream_scope;
        if (p.graph_enabled) {
            if (!p.graph_stream) p.graph_stream = c10::cuda::getStreamFromPool(false, 0);
            stream_scope.emplace(*p.graph_stream);
        }
        int64_t fd[] = {1, frames, 768}, pd[] = {1, frames}, one[] = {1}, sid = 0;
        auto input = [&](const void* data, const int64_t* shape, size_t rank, int type) {
            if (p.graph_enabled)
                return at::from_blob(const_cast<void*>(data), at::IntArrayRef(shape, rank), at::TensorOptions().dtype(dtype(type)));
            return tensor(data, shape, rank, type, true, true);
        };
        std::vector<torch::jit::IValue> args = {
            input(features, fd, 3, 0), input(&frames, one, 1, 2),
            input(pitch, pd, 2, 2), input(pitchf, pd, 2, 0),
            input(&sid, one, 1, 2), skip, ret, formant};
        torch::jit::GraphOptimizerEnabledGuard guard(false);
        at::Tensor result;
        if (!p.graph_enabled) {
            result = p.module.get_method("infer")(args).toTuple()->elements()[0].toTensor()[0][0];
        } else {
            if (!p.cuda_graph) {
                auto cpu_gen = at::detail::getDefaultCPUGenerator();
                auto gpu_gen = at::cuda::detail::getDefaultCUDAGenerator(0);
                auto cpu_state = cpu_gen.get_state();
                auto gpu_state = gpu_gen.get_state();
                auto new_graph = std::make_unique<at::cuda::CUDAGraph>();
                auto new_args = args;
                at::Tensor new_output;
                try {
                    c10::cuda::device_synchronize();
                    for (size_t i = 0; i < 5; ++i)
                        new_args[i] = args[i].toTensor().to(at::Device(at::kCUDA, 0), args[i].toTensor().scalar_type(), false, true);
                    for (int i = 0; i < 3; ++i) p.module.get_method("infer")(new_args);
                    c10::cuda::device_synchronize();
                    cpu_gen.set_state(cpu_state);
                    gpu_gen.set_state(gpu_state);
                    new_graph->capture_begin();
                    try {
                        new_output = p.module.get_method("infer")(new_args).toTuple()->elements()[0].toTensor()[0][0];
                        new_output.clamp_(-1., 1.);
                    } catch (...) {
                        try { new_graph->capture_end(); } catch (...) {}
                        throw;
                    }
                    new_graph->capture_end();
                    c10::cuda::device_synchronize();
                    cpu_gen.set_state(cpu_state);
                    gpu_gen.set_state(gpu_state);
                    TORCH_CHECK(new_output.scalar_type() == at::kHalf && static_cast<size_t>(new_output.numel()) == output_len,
                        "captured output dtype/length mismatch");
                    p.static_args = std::move(new_args);
                    p.static_output = std::move(new_output);
                    p.graph_output_len = static_cast<size_t>(p.static_output.numel());
                    p.cuda_graph = std::move(new_graph);
                } catch (...) {
                    p.poisoned = true;
                    try { new_graph->reset(); c10::cuda::device_synchronize(); } catch (...) {}
                    cpu_gen.set_state(cpu_state);
                    gpu_gen.set_state(gpu_state);
                    throw;
                }
            }
            TORCH_CHECK(output_len == p.graph_output_len, "graph output size changed; recreate engine");
            for (size_t i = 5; i < 8; ++i)
                TORCH_CHECK(args[i].toInt() == p.static_args[i].toInt(), "graph startup changed; recreate engine");
            for (size_t i = 0; i < 5; ++i) {
                auto target = p.static_args[i].toTensor();
                auto source = args[i].toTensor();
                TORCH_CHECK(target.sizes() == source.sizes() && target.scalar_type() == source.scalar_type(),
                    "graph input shape/dtype changed; recreate engine");
                target.copy_(source, false);
            }
            p.cuda_graph->replay();
            result = p.static_output;
        }
        TORCH_CHECK(result.scalar_type() == at::kHalf, "native output is not fp16");
        if (!p.graph_enabled) result.clamp_(-1., 1.);
        auto host = result.to(at::kFloat).cpu().contiguous();
        TORCH_CHECK(static_cast<size_t>(host.numel()) == output_len, "output length mismatch");
        TORCH_CHECK(at::isfinite(host).all().item<bool>(), "non-finite output");
        std::memcpy(output, host.data_ptr<float>(), output_len * sizeof(float));
    });
    if (code != 0) static_cast<Probe*>(ptr)->poisoned = true;
    return code;
}
}
