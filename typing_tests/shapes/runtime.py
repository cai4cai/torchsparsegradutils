"""Runtime counterparts of the shape contracts; no shape stubs are imported."""

import ast
import inspect
from pathlib import Path

import torch

import torchsparsegradutils as tsgu
from torchsparsegradutils.utils.random_sparse import (
    generate_random_sparse_coo_matrix,
    generate_random_sparse_csr_matrix,
)


def check_matmul() -> None:
    torch.manual_seed(7)
    for make, require, is_layout, layout in (
        (generate_random_sparse_coo_matrix, tsgu.require_sparse_coo, tsgu.is_sparse_coo, torch.sparse_coo),
        (generate_random_sparse_csr_matrix, tsgu.require_sparse_csr, tsgu.is_sparse_csr, torch.sparse_csr),
    ):
        for size in ((3, 4), (2, 3, 4)):
            a = make(size, 5, values_dtype=torch.float64).requires_grad_()
            assert require(a) is a
            assert type(a) is torch.Tensor
            assert is_layout(a)
            b = torch.randn(*size[:-2], 4, 2, dtype=torch.float64, requires_grad=True)
            result = tsgu.sparse_mm(a, b)
            torch.testing.assert_close(result, a.to_dense() @ b)
            result.sum().backward()
            assert a.grad is not None and a.grad.layout == layout
            assert b.grad is not None and b.grad.shape == b.shape
            # PyTorch cannot convert a 3-sparse-dimension COO batch to CSC.
            if len(size) == 2:
                assert a.to_sparse_csc().shape == a.shape
            assert a.to_dense().shape == a.shape
        try:
            require(torch.randn(3, 4))
        except TypeError:
            pass
        else:
            raise AssertionError("Layout validator accepted a dense tensor")


def check_solvers() -> None:
    a = tsgu.require_sparse_csr(torch.eye(4, dtype=torch.float64).to_sparse_csr())
    b = torch.randn(4, 2, dtype=torch.float64)
    torch.testing.assert_close(tsgu.sparse_triangular_solve(a, b), b)
    torch.testing.assert_close(tsgu.sparse_generic_solve(a, b), b)
    torch.testing.assert_close(tsgu.sparse_generic_solve(a, b[:, 0]), b[:, 0])
    rectangular = torch.cat((torch.eye(3, dtype=torch.float64), torch.zeros(2, 3, dtype=torch.float64)))
    sparse = tsgu.require_sparse_coo(rectangular.to_sparse_coo())
    rhs = torch.randn(5, 2, dtype=torch.float64)
    torch.testing.assert_close(tsgu.sparse_generic_lstsq(sparse, rhs), rhs[:3])
    torch.testing.assert_close(tsgu.sparse_generic_lstsq(sparse, rhs[:, 0]), rhs[:3, 0])


def check_stub_parameters() -> None:
    # Guard against accidental stub drift in public function parameter names,
    # positional/keyword kinds, and optionality. Types are checked by Pyrefly.
    import importlib

    stub_root = Path(tsgu.__file__).parent / "_shape_stubs" / "torchsparsegradutils-stubs"
    for path in sorted(stub_root.rglob("*.pyi")):
        module_name = "torchsparsegradutils." + ".".join(path.relative_to(stub_root).with_suffix("").parts)
        module = importlib.import_module(module_name)
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.FunctionDef):
                continue
            signature = inspect.signature(getattr(module, node.name))
            args = node.args
            names = [x.arg for x in args.posonlyargs + args.args + args.kwonlyargs]
            if args.vararg:
                names.append(args.vararg.arg)
            if args.kwarg:
                names.append(args.kwarg.arg)
            # An overload may omit an optional argument to select its default,
            # or require an optional argument to select an explicit value.
            assert set(names).issubset(signature.parameters), (module_name, node.name, names)
            for name, parameter in signature.parameters.items():
                assert name in names or parameter.default is not inspect.Parameter.empty, (module_name, node.name, name)
            groups = (
                (args.posonlyargs, inspect.Parameter.POSITIONAL_ONLY),
                (args.args, inspect.Parameter.POSITIONAL_OR_KEYWORD),
                (args.kwonlyargs, inspect.Parameter.KEYWORD_ONLY),
                ([args.vararg] if args.vararg else [], inspect.Parameter.VAR_POSITIONAL),
                ([args.kwarg] if args.kwarg else [], inspect.Parameter.VAR_KEYWORD),
            )
            for arguments, kind in groups:
                for argument in arguments:
                    assert signature.parameters[argument.arg].kind == kind, (module_name, node.name, argument.arg)
            positional = args.posonlyargs + args.args
            defaulted = positional[len(positional) - len(args.defaults) :] if args.defaults else []
            defaulted += [arg for arg, default in zip(args.kwonlyargs, args.kw_defaults) if default is not None]
            for argument in defaulted:
                assert signature.parameters[argument.arg].default is not inspect.Parameter.empty, (
                    module_name,
                    node.name,
                    argument.arg,
                )


if __name__ == "__main__":
    check_matmul()
    check_solvers()
    check_stub_parameters()
    print("Runtime shape contracts passed: identity, layouts, outputs, gradients, and public stub parameters.")
