import ast
import importlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACTS_ROOT = REPO_ROOT / "Python" / "src" / "hexorl" / "contracts"

EXPECTED_CONTRACT_MODULES = {
    "hexorl.contracts.identity": (),
    "hexorl.contracts.history": ("MoveHistory",),
    "hexorl.contracts.coordinates": (),
    "hexorl.contracts.symmetry": (
        "apply_tensor_symmetry",
        "compose_symmetries",
        "inverse_symmetry",
        "transform_axis_label",
        "transform_axis_maps",
        "transform_dense_policy",
        "transform_history",
        "transform_legal_table",
        "transform_pair_policy_target",
        "transform_policy_target",
        "transform_qr",
    ),
    "hexorl.contracts.legal": ("LegalActionTable",),
    "hexorl.contracts.actions": (),
    "hexorl.contracts.targets": ("PairPolicyTarget", "PolicyTarget"),
    "hexorl.contracts.tactical": (),
    "hexorl.contracts.candidates": ("CandidateTable",),
    "hexorl.contracts.pairs": ("PairActionTable",),
    "hexorl.contracts.graph": (),
    "hexorl.contracts.replay": (),
    "hexorl.contracts.telemetry": ("ContractTrace",),
    "hexorl.contracts.validation": (),
    "hexorl.contracts.debug": (),
}

FORBIDDEN_CONTRACT_IMPORT_PREFIXES = (
    "hexorl.dashboard",
    "hexorl.inference",
    "hexorl.model",
    "hexorl.search",
    "hexorl.selfplay",
    "hexorl.train",
    "hexorl.tuning",
)


def _imports_from(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def test_phase01_contract_modules_and_public_symbols_exist():
    importlib.import_module("hexorl.contracts")

    missing_symbols: list[str] = []
    for module_name, public_symbols in EXPECTED_CONTRACT_MODULES.items():
        module = importlib.import_module(module_name)
        for symbol in public_symbols:
            if not hasattr(module, symbol):
                missing_symbols.append(f"{module_name}.{symbol}")

    assert missing_symbols == []


def test_contracts_do_not_import_runtime_orchestration_packages():
    assert CONTRACTS_ROOT.is_dir(), "Phase 01 contracts package is required"

    violations: list[str] = []
    for path in sorted(CONTRACTS_ROOT.rglob("*.py")):
        for imported in _imports_from(path):
            if imported.startswith(FORBIDDEN_CONTRACT_IMPORT_PREFIXES):
                relative = path.relative_to(REPO_ROOT)
                violations.append(f"{relative}: imports {imported}")

    assert violations == []
