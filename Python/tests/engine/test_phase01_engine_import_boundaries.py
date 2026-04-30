import ast
import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_SRC = REPO_ROOT / "Python" / "src"
HEXORL_ROOT = PYTHON_SRC / "hexorl"
ENGINE_ROOT = HEXORL_ROOT / "engine"

EXPECTED_ENGINE_MODULES = (
    "hexorl.engine",
    "hexorl.engine.rust",
    "hexorl.engine.legal",
    "hexorl.engine.history",
    "hexorl.engine.encoding",
    "hexorl.engine.parity",
)

EXPECTED_ENGINE_SYMBOLS = {
    "hexorl.engine.legal": ("LegalTableProvider",),
}

DIRECT_ENGINE_IMPORTS = {"_engine"}
FIXTURE_TOOLING_NAMES = {"fixture", "fixtures"}


def _imports_from(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def _is_fixture_tooling(path: Path) -> bool:
    names = {part.lower() for part in path.relative_to(HEXORL_ROOT).parts}
    return bool(names & FIXTURE_TOOLING_NAMES)


def test_phase01_engine_modules_and_public_symbols_exist():
    for module_name in EXPECTED_ENGINE_MODULES:
        importlib.import_module(module_name)

    missing_symbols: list[str] = []
    for module_name, public_symbols in EXPECTED_ENGINE_SYMBOLS.items():
        module = importlib.import_module(module_name)
        for symbol in public_symbols:
            if not hasattr(module, symbol):
                missing_symbols.append(f"{module_name}.{symbol}")

    assert missing_symbols == []


def test_runtime_code_imports_engine_boundary_not_private_pyo3_module():
    assert ENGINE_ROOT.is_dir(), "Phase 01 engine package is required"

    violations: list[str] = []
    for path in sorted(HEXORL_ROOT.rglob("*.py")):
        if ENGINE_ROOT in path.parents or _is_fixture_tooling(path):
            continue
        direct_imports = DIRECT_ENGINE_IMPORTS.intersection(_imports_from(path))
        if direct_imports:
            relative = path.relative_to(REPO_ROOT)
            violations.append(f"{relative}: imports {', '.join(sorted(direct_imports))}")

    assert violations == []
