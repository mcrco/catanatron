from setuptools import Extension, setup

try:
    import pybind11
except ImportError as exc:  # pragma: no cover - exercised by build tooling
    raise RuntimeError(
        "pybind11 is required to build the C++ Catanatron extension"
    ) from exc


ext_modules = [
    Extension(
        "catanatron._cpp_engine",
        ["cpp/src/core.cpp", "cpp/python/bindings.cpp"],
        include_dirs=["cpp/include", pybind11.get_include()],
        language="c++",
        extra_compile_args=["-std=c++17"],
    )
]


setup(ext_modules=ext_modules)
