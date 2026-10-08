"""The released package as a package: what a fresh clone can import and run before it has any data."""
import ast
import importlib
import inspect
import pathlib

from helpers import transit_network

import endogenous_sue


def test_every_deferred_import_in_the_package_resolves():
    """Function-local imports are invisible to importing the package, so they need their own check.

    They exist to break import cycles, they raise only when the branch is taken, and none is reached by a
    test that stops at module import.

    A try and except pair around an import names one capability reached by two spellings, so neither
    branch is individually required: the handler runs only because the body failed. What is required is
    that the pair resolves somehow, and that is checked as a disjunction. Requiring the fallback spelling
    to resolve on a machine that never takes it would fail for the wrong reason.
    """
    root = pathlib.Path(endogenous_sue.__file__).parent
    unresolved = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        package = root.name + "".join("." + part for part in path.relative_to(root).parts[:-1])

        def failure(node, package=package, path=path):
            if isinstance(node, ast.ImportFrom):
                target = ("." * node.level) + (node.module or "")
                names = [alias.name for alias in node.names]
            else:
                target, names = None, [alias.name for alias in node.names]
            for name in ([target] if target else names):
                try:
                    module = importlib.import_module(name, package if target else None)
                except ImportError as exc:
                    return f"{path.name}:{node.lineno} {name}: {exc}"
                if target:
                    for attribute in names:
                        if not hasattr(module, attribute):
                            return f"{path.name}:{node.lineno} {name}.{attribute} is missing"
            return None

        guarded = {}
        optional = set()
        for guard in (node for node in ast.walk(tree) if isinstance(node, ast.Try)):
            caught = {getattr(name, "id", None)
                      for handler in guard.handlers
                      for name in ast.walk(handler.type) if handler.type is not None}
            if not (caught & {"ImportError", "ModuleNotFoundError", "Exception"}
                    or any(handler.type is None for handler in guard.handlers)):
                continue
            body_imports = [node for statement in guard.body for node in ast.walk(statement)
                            if isinstance(node, (ast.Import, ast.ImportFrom))]
            handler_imports = [node for handler in guard.handlers for statement in handler.body
                               for node in ast.walk(statement)
                               if isinstance(node, (ast.Import, ast.ImportFrom))]
            if body_imports and not handler_imports:
                optional |= {id(node) for node in body_imports}
                continue
            guarded[id(guard)] = body_imports + handler_imports

        skip = {id(node) for node in tree.body} | optional
        for spellings in guarded.values():
            skip |= {id(node) for node in spellings}
            reasons = [failure(node) for node in spellings]
            if spellings and all(reasons):
                unresolved.append(" and ".join(reason for reason in reasons if reason))

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)) and id(node) not in skip:
                reason = failure(node)
                if reason:
                    unresolved.append(reason)
    assert not unresolved, "deferred imports that do not resolve:\n  " + "\n  ".join(unresolved)


def test_no_module_reads_a_solver_setting_from_the_environment():
    """A published result must not depend on a shell.

    Only the location of the benchmark data may be set that way, because it says where the inputs are and
    not what is computed from them.
    """
    root = pathlib.Path(endogenous_sue.__file__).parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            attribute = node.func
            reads_environment = (
                attribute.attr == "getenv"
                or (attribute.attr in {"get", "setdefault"}
                    and isinstance(attribute.value, ast.Attribute)
                    and attribute.value.attr == "environ"))
            if not reads_environment:
                continue
            named = [argument.value for argument in node.args if isinstance(argument, ast.Constant)]
            if named != ["ENDOGENOUS_SUE_DATA"]:
                offenders.append(f"{path.name}:{node.lineno} reads {named}")
    assert not offenders, "solver settings read from the environment:\n  " + "\n  ".join(offenders)


def test_the_solver_obeys_the_transit_rule_its_own_support_applies():
    """A network whose cheapest route crosses a foreign centroid must not lose its demand.

    The transit rule forbids that crossing, so the support deletes the links into the foreign centroid. A
    potential built without the same restriction prices a route the support has removed, the only link
    leaving the origin then fails strict decrease, and the origin injects nothing while the flow residual
    stays at machine precision. Synthetic, because the corpus networks that expose this are large.
    """
    from endogenous_sue.certificate import certify, conservation_error
    from endogenous_sue.equilibrium import solve
    from endogenous_sue.network import NetworkArrays

    net, od = transit_network()
    topo = NetworkArrays.from_dict(net).topology()
    assert topo.no_thru and not topo.zero_dep_conn      # the combination an inferred rule gets wrong

    equilibrium = solve(net, od, 1.0)
    assert conservation_error(net, od, equilibrium.x)[0] < 1e-8
    assert equilibrium.x[2] > 9.99                      # all the demand takes the legal route
    assert certify(net, od, 1.0, equilibrium.x).certified


def test_the_potential_cannot_be_built_under_a_different_rule_than_the_support():
    """The transit rule is read from the topology and cannot be supplied separately."""
    from endogenous_sue.shortestpath import potentials

    parameters = list(inspect.signature(potentials).parameters)
    assert parameters == ["topo", "cost"], (
        "potentials() must take only a topology and a cost, so the transit rule cannot disagree with "
        f"the one the support applies; it takes {parameters}")
