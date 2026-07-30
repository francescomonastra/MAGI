"""Every name in magi.__all__ must carry a hover-usable docstring.

VSCode (and any Jedi/Pylance-backed editor) shows a symbol's docstring on
hover, so for MAGI's users that docstring *is* the API reference - there is no
separate generated doc site. A one-line summary is not enough: the thing a
caller actually needs at the call site is what each argument means and what
comes back. This test pins that contract so it cannot silently rot as the
public surface grows.

Rules enforced, per exported symbol:
  1. It has a docstring at all.
  2. If it takes arguments, the docstring has a "Parameters" section
     (NumPy style, matching the rest of the package).
  3. If it is a function, the docstring has a "Returns" section - including
     for the report_*/print_* helpers, where "Returns None" is the useful
     answer rather than an omission.

Keyword-only functions (save_final_trained_model, the export entry points)
have no positional args, so rule 2 is checked against the full argument list
including kwonly.
"""
import ast
import pathlib
import re

import pytest

PKG = pathlib.Path(__file__).resolve().parents[1] / "magi"

PARAM_SECTION = re.compile(r"Parameters\s*\n\s*-{3,}", re.M)
RETURN_SECTION = re.compile(r"Returns\s*\n\s*-{3,}", re.M)

# Private helper re-exported for notebook convenience; not part of the
# documented surface a user is expected to call.
EXEMPT = {"_save_and_show"}


def _exported_names():
    tree = ast.parse((PKG / "__init__.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "__all__" for t in node.targets
        ):
            return [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
    raise AssertionError("magi/__init__.py defines no __all__")


def _top_level_defs():
    """Map name -> (path, ast node) for every top-level def/class in the package."""
    out = {}
    for path in sorted(PKG.rglob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                out.setdefault(node.name, (path, node))
    return out


def _top_level_bindings():
    """Names bound by a module-level assignment anywhere in the package.

    Covers the two kinds of export that are not a def or a class: the
    backward-compatible aliases in config (`set_seed`,
    `configure_tensorflow`) and data constants such as
    DEFAULT_CANDIDATE_ENERGY_LINES. These resolve, but a plain assignment
    carries no docstring for hover to show, so they are checked for
    existence only.
    """
    out = set()
    for path in sorted(PKG.rglob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        out.add(t.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                out.add(node.target.id)
    return out


DEFS = _top_level_defs()
BINDINGS = _top_level_bindings()
EXPORTED = [
    n for n in _exported_names()
    if not n.startswith("__") and n not in EXEMPT and n in DEFS
]


def test_exported_names_are_resolvable():
    """__all__ must not advertise a name the package does not define.

    Guards the test below from silently skipping: it filters on `n in DEFS`,
    so a typo'd export would otherwise vanish from coverage rather than fail.
    A name bound by a module-level assignment (alias or constant) resolves
    fine but has no docstring, so it passes here and is not hover-checked.
    """
    missing = [
        n for n in _exported_names()
        if not n.startswith("__")
        and n not in EXEMPT
        and n not in DEFS
        and n not in BINDINGS
    ]
    assert not missing, (
        "magi.__all__ exports names with no top-level definition or "
        f"assignment in the package: {missing}"
    )


@pytest.mark.parametrize("name", EXPORTED)
def test_public_symbol_is_documented_for_hover(name):
    path, node = DEFS[name]
    where = f"{path.relative_to(PKG.parent)}:{node.lineno} ({name})"

    doc = ast.get_docstring(node)
    assert doc, f"{where}: exported symbol has no docstring"

    if isinstance(node, ast.FunctionDef):
        args = [
            a.arg
            for a in (node.args.posonlyargs + node.args.args + node.args.kwonlyargs)
            if a.arg not in ("self", "cls")
        ]
        if args:
            assert PARAM_SECTION.search(doc), (
                f"{where}: takes {len(args)} argument(s) {args[:6]} but the "
                "docstring has no NumPy-style 'Parameters\\n----------' "
                "section, so hover will not tell a caller what to pass"
            )
        assert RETURN_SECTION.search(doc), (
            f"{where}: no 'Returns\\n-------' section. For a print_*/report_* "
            "helper document it explicitly as returning None rather than "
            "leaving it out"
        )
