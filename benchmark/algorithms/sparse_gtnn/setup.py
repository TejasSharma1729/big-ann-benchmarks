from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import os
import sys
import subprocess

class BuildExt(build_ext):
    def run(self):
        # 1. Provide a hint to not skip the build if it's already done,
        #    so we are sure the library is there for stubgen? 
        #    Actually, standard run() checks timestamps.
        super().run()
        
        # 2. Generate the .pyi file using stubgen
        # We need to run stubgen on the *built* extension.
        # self.build_lib contains the directory where the .so/.dylib file was generated.
        print("Generating type stubs (stubgen)...")
        
        # Helper to find where the module might be located
        build_lib = os.path.abspath(self.build_lib)
        
        # Add the build dir to PYTHONPATH so stubgen can import the new compiled module
        env = os.environ.copy()
        env["PYTHONPATH"] = build_lib + os.pathsep + env.get("PYTHONPATH", "")

        # Run stubgen as a subprocess
        # -m sparse_gtnn: the module we just built
        # -o build_lib: output directly next to the .so file
        try:
            subprocess.check_call(
                [sys.executable, "-m", "mypy.stubgen", "-m", "sparse_gtnn", "-o", build_lib],
                env=env
            )
            print(f"Stub file generated in {build_lib}")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to generate stubs: {e}")

ext_modules = [
    Extension(
        "sparse_gtnn",
        sorted([
            "src/pybind11_module.cpp",
            # Add other cpp files if any. Currently only pybind11_module.cpp seems to link everything (header-only lib approach?)
            # Let's check pybind11_module.cpp includes.
        ]),
        include_dirs=[
            pybind11.get_include(),
            "src" 
        ],
        extra_compile_args=["-O3", "-march=native", "-ffast-math", "-std=c++20", "-pthread", "-funroll-loops", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        language="c++",
    ),
]

setup(
    name="sparse_gtnn",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)