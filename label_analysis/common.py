from pathlib import Path
import sys

# Resolve path to the compiled .so in cpp/build
_cpp_path = Path(__file__).parent / "cpp" / "build"
if _cpp_path.exists():
    sys.path.insert(0, str(_cpp_path))

try:
    import printcpp
    _HAS_CPP = True
except ImportError:
    print("printcpp module not found — run CMake first.")
    _HAS_CPP = False
