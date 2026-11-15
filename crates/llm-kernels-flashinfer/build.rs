fn main() {
    let dst = cmake::Config::new("native").build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());

    println!("cargo:rustc-link-lib=dylib=flashinfer_adapter");
    println!("cargo:rustc-link-lib=cudart");

    println!("cargo:rerun-if-changed=native/CMakeLists.txt");
    println!("cargo:rerun-if-changed=native/include");
    println!("cargo:rerun-if-changed=native/src");
}
