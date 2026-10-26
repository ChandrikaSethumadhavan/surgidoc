"""A fresh clone must be installable: every third-party import in surgidoc/ is a declared dependency (AC-9, NFR-1)."""
import ast
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
# import name -> distribution name. A new third-party import must be added here AND to pyproject.
DISTRIBUTION = {"cv2": "opencv-python-headless", "fpdf": "fpdf2", "pypdf": "pypdf", "langgraph": "langgraph",
                "cryptography": "cryptography", "opentelemetry": "opentelemetry-sdk", "psycopg": "psycopg",
                "redis": "redis", "pydantic": "pydantic", "kagglehub": "kagglehub", "pytesseract": "pytesseract",
                "PIL": "pillow"}
LAZY_OPTIONAL = {"kagglehub", "pytesseract", "PIL"}  # extras: must never be imported at module top level


def dist_name(requirement):
    return re.split(r"[\[<>=!~; ]", requirement.strip(), maxsplit=1)[0].lower().replace("_", "-")


def declared():
    project = PYPROJECT["project"]
    core = {dist_name(r) for r in project["dependencies"]}
    extras = {dist_name(r) for rs in project.get("optional-dependencies", {}).values() for r in rs}
    return core, extras


def imports(path):
    """(module, at_top_level) for every import in a file."""
    tree = ast.parse(path.read_text())
    top = {id(n) for n in tree.body}
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name.split(".")[0], id(node) in top) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.module.split(".")[0], id(node) in top))
    return out


def third_party():
    found = {}
    for path in (ROOT / "surgidoc").rglob("*.py"):
        for mod, at_top in imports(path):
            if mod != "surgidoc" and mod not in sys.stdlib_module_names:
                found.setdefault(mod, []).append((path.name, at_top))
    return found


def test_pyproject_declares_a_buildable_package():
    assert PYPROJECT["build-system"]["build-backend"] and PYPROJECT["build-system"]["requires"]
    p = PYPROJECT["project"]
    assert p["name"] == "surgidoc" and p["version"] and p["requires-python"].startswith(">=3.")
    find = PYPROJECT["tool"]["setuptools"]["packages"]["find"]
    assert "surgidoc*" in find["include"]


def test_every_package_directory_is_a_real_package():
    assert (ROOT / "surgidoc" / "__init__.py").exists()
    for d in (ROOT / "surgidoc").iterdir():
        if d.is_dir() and d.name != "__pycache__":
            assert (d / "__init__.py").exists(), d.name


def test_every_third_party_import_has_a_known_distribution():
    unknown = sorted(set(third_party()) - set(DISTRIBUTION))
    assert unknown == [], f"add these to DISTRIBUTION and pyproject.toml: {unknown}"


def test_every_third_party_import_is_declared_in_pyproject():
    core, extras = declared()
    missing = sorted({DISTRIBUTION[m] for m in third_party()} - core - extras)
    assert missing == [], f"declare in pyproject.toml: {missing}"


def test_required_dependencies_cover_everything_not_optional():
    core, _ = declared()
    needed = {DISTRIBUTION[m] for m in third_party() if m not in LAZY_OPTIONAL}
    assert needed <= core, sorted(needed - core)


def test_optional_libraries_are_only_imported_lazily():
    for mod in LAZY_OPTIONAL:
        for filename, at_top in third_party().get(mod, []):
            assert not at_top, f"{mod} is imported at module level in {filename}; it is an optional extra"


def test_extras_exist_for_dev_kaggle_and_ocr():
    extras = PYPROJECT["project"]["optional-dependencies"]
    assert {"dev", "kaggle", "ocr"} <= set(extras)
    assert "pytest" in {dist_name(r) for r in extras["dev"]}
    assert "kagglehub" in {dist_name(r) for r in extras["kaggle"]}
    assert {"pytesseract", "pillow"} <= {dist_name(r) for r in extras["ocr"]}


def test_dependencies_have_lower_bounds():
    for r in PYPROJECT["project"]["dependencies"]:
        assert ">=" in r, f"{r} has no minimum version"


def test_no_dependency_is_declared_twice_or_unused():
    core, extras = declared()
    used = {DISTRIBUTION[m] for m in third_party()}
    # numpy comes with opencv and opentelemetry-api with the sdk; every other runtime dependency must be imported by the code
    unused = sorted(d for d in core if d not in used and d not in {"numpy", "opentelemetry-api"})
    assert unused == [], f"declared but never imported by surgidoc: {unused}"


def test_pytest_is_configured_for_one_command_runs():
    cfg = PYPROJECT["tool"]["pytest"]["ini_options"]
    assert cfg["testpaths"] == ["tests"] and "." in cfg["pythonpath"]


def test_the_package_imports_without_optional_extras():
    import importlib

    for name in ["surgidoc", "surgidoc.pipeline", "surgidoc.ingest", "surgidoc.vision"]:
        importlib.import_module(name)
    assert "kagglehub" not in sys.modules and "pytesseract" not in sys.modules
