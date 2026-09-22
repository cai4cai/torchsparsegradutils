"""Require review of inherited Self annotations when upgrading the torch overlay."""

import argparse
import ast
import io
import tokenize
from pathlib import Path

import torchsparsegradutils

# Only the scalar overloads of these operators may return Self. Their tensor
# overloads must compute a broadcast shape, as checked by the static contracts.
SCALAR_OPERATORS = {
    "__add__",
    "__sub__",
    "__mul__",
    "__mod__",
    "__truediv__",
    "__floordiv__",
    "__radd__",
    "__rsub__",
    "__rmul__",
    "__rtruediv__",
    "__rpow__",
    "__eq__",
    "__ne__",
    "__and__",
    "__or__",
    "__xor__",
    "__rand__",
    "__ror__",
    "__rxor__",
}

# Reviewed shape-preserving signatures. This is not a promise that PyTorch has
# a sparse kernel for every method/layout/dtype combination. In-place writes
# and arbitrary alias mutation are outside these static shape contracts.
SHAPE_PRESERVING = {
    "__neg__",
    "__abs__",
    "__invert__",
    "zero_",
    "add_",
    "pin_memory",
    "byte",
    "contiguous",
    "clone",
    "detach",
    "float",
    "half",
    "double",
    "int",
    "long",
    "bool",
    "to",
    "type_as",
    "cuda",
    "cpu",
    "copy_",
    "fill_",
    "requires_grad_",
    "tril",
    "triu",
    "neg",
    "abs",
    "floor",
    "ceil",
    "round",
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "sqrt",
    "tanh",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "exp2",
    "expm1",
    "log2",
    "log10",
    "log1p",
    "rsqrt",
    "square",
    "reciprocal",
    "sign",
    "sigmoid",
    "trunc",
    "frac",
    "logical_not",
    "relu",
    "erf",
    "erfc",
    "erfinv",
    "lgamma",
    "digamma",
    "polygamma",
    "asinh",
    "acosh",
    "atanh",
    "deg2rad",
    "deg2rad_",
    "rad2deg",
    "bitwise_not",
    "isreal",
    "isposinf",
    "isneginf",
    "isnan",
    "isinf",
    "isfinite",
    "cholesky",
    "inverse",
    "matrix_power",
    "masked_fill_",
    "masked_scatter_",
    "index_add",
    "index_add_",
    "index_copy",
    "index_copy_",
    "index_put",
    "index_put_",
    "index_fill",
    "index_fill_",
    "put",
    "put_",
    "bernoulli",
    "bernoulli_",
    "normal_",
    "random_",
    "uniform_",
}


def tensor_members(path: Path) -> tuple[list[tuple[str, list[str]]], set[str]]:
    # Tokenization can read upstream's Python 3.13 syntax on Python 3.12 without
    # importing the stubs or rewriting them to fit the runtime AST parser.
    tokens = list(tokenize.generate_tokens(io.StringIO(path.read_text()).readline))
    start = next(
        i for i, token in enumerate(tokens[:-1]) if token.string == "class" and tokens[i + 1].string == "Tensor"
    )
    start = next(i for i in range(start, len(tokens)) if tokens[i].type == tokenize.INDENT)
    depth = 0
    methods = []
    properties = set()
    for i in range(start, len(tokens)):
        token = tokens[i]
        if token.type == tokenize.INDENT:
            depth += 1
        elif token.type == tokenize.DEDENT:
            depth -= 1
            if depth == 0:
                break
        if depth != 1 or token.type != tokenize.NAME:
            continue
        if token.string != "def":
            if (
                token.start[1] == len(tokens[start].string)
                and tokens[i + 1].string == ":"
                and tokens[i + 2].string == "Self"
            ):
                properties.add(token.string)
            continue
        name = tokens[i + 1].string
        signature = []
        nesting = 0
        for part in tokens[i + 2 :]:
            if part.string == ":" and nesting == 0:
                break
            if part.string in ("(", "[", "{"):
                nesting += 1
            elif part.string in (")", "]", "}"):
                nesting -= 1
            if part.type not in (tokenize.NL, tokenize.COMMENT):
                signature.append(part.string)
        if "->" in signature:
            arrow = signature.index("->")
            if "Self" in signature[arrow + 1 :]:
                methods.append((name, signature[:arrow]))
    assert methods and properties, "Could not locate upstream Tensor Self annotations"
    return methods, properties


def check_inherited_self(path: Path) -> None:
    overlay = Path(torchsparsegradutils.__file__).parent / "_shape_stubs" / "torchsparsegradutils-stubs"
    tree = ast.parse((overlay / "sparse_types.pyi").read_text())
    sparse = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_SparseTensor")
    overrides = {node.name for node in sparse.body if isinstance(node, ast.FunctionDef)}
    methods, properties = tensor_members(path)
    for name, parameters in methods:
        if name in overrides:
            continue
        if name in SCALAR_OPERATORS:
            assert "Tensor" not in parameters, f"Review tensor overload returning Self: {name}"
        else:
            assert name in SHAPE_PRESERVING, f"Review inherited Self method: {name}"
    assert properties <= overrides | {"real", "imag", "data"}, properties - overrides
    print(f"Audited {len(methods)} inherited Self signatures and {len(properties)} properties.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("torch_stub", type=Path, help="Path to torch-stubs/__init__.pyi")
    check_inherited_self(parser.parse_args().torch_stub)
