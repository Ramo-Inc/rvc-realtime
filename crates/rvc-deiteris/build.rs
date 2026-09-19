use std::path::PathBuf;

fn main() {
    if std::env::var_os("CARGO_FEATURE_NATIVE").is_none() { return; }
    println!("cargo:rerun-if-env-changed=RVC_LIBTORCH");
    println!("cargo:rerun-if-changed=bridge.cpp");
    println!("cargo:rerun-if-env-changed=RVC_CUDA_TOOLKIT");
    let cuda_toolkit = std::env::var_os("RVC_CUDA_TOOLKIT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
        });
    let root = PathBuf::from(
        std::env::var_os("RVC_LIBTORCH")
            .expect("Set RVC_LIBTORCH to the Torch 2.7.1 directory containing include/ and lib/"),
    );
    cc::Build::new()
        .cpp(true)
        .file("bridge.cpp")
        .include(root.join("include"))
        .include(root.join("include/torch/csrc/api/include"))
        .include(cuda_toolkit.join("include"))
        .flag("/std:c++17")
        .flag("/EHsc")
        .flag("/bigobj")
        .warnings(false)
        .compile("deiteris_jit_bridge");
    println!(
        "cargo:rustc-link-search=native={}",
        root.join("lib").display()
    );
    for name in ["torch", "torch_cpu", "torch_cuda", "c10", "c10_cuda"] {
        println!("cargo:rustc-link-lib=dylib={name}");
    }
    // CUDAStreamGuard imports CUDA 12 APIs; runtime uses the staged Torch DLL.
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_toolkit.join("lib/x64").display()
    );
    println!("cargo:rustc-link-lib=dylib=cudart");
}
