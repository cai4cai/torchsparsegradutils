# Documentation Build Instructions

## Building Locally

### Prerequisites

Install the documentation dependencies:

```bash
uv sync --locked --group docs
```

Or install from requirements file:

```bash
uv sync --locked --group docs
```

### Building HTML Documentation

```bash
cd docs/
make html
```

The documentation will be built in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser.

### Building PDF Documentation

```bash
cd docs/
make latexpdf
```

### Live Rebuild (Development)

For automatic rebuilding during development:

```bash
cd docs/
sphinx-autobuild source _build/html
```

This will serve the documentation at http://localhost:8000 and automatically rebuild when files change.

### Cleaning Build Files

```bash
cd docs/
make clean
```

## Read the Docs Integration

This project is configured for [Read the Docs](https://readthedocs.org) automatic building:

- Configuration: `.readthedocs.yaml`
- Builds are triggered on git pushes
- Multiple formats: HTML, PDF, ePub
- Documentation dependencies live in the `docs` dependency-group of `pyproject.toml` (uv-managed)

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py              # Sphinx configuration
│   ├── index.rst            # Main documentation page
│   ├── installation.rst     # Installation guide
│   ├── quickstart.rst       # Quick start guide
│   ├── benchmarks.rst       # Performance benchmarks
│   ├── contributing.rst     # Contributing guide
│   ├── api/                 # API reference
│   │   ├── index.rst
│   │   ├── core.rst         # Core functions
│   │   ├── utils.rst        # Utility functions
│   │   ├── distributions.rst # Probability distributions
│   │   ├── encoders.rst     # Encoder utilities
│   │   └── backends.rst     # Backend integrations
│   ├── tutorials/           # Detailed tutorials
│   │   ├── index.rst
│   │   ├── basic_operations.rst
│   │   ├── linear_solvers.rst
│   │   ├── distributions.rst
│   │   ├── backends.rst
│   │   └── optimization_examples.rst
│   └── _static/             # Static assets (CSS, images)
│       └── custom.css       # Custom styling
├── Makefile                 # Build commands (Unix/Mac)
├── make.bat                 # Build commands (Windows)
```

## Writing Documentation

### Adding New Pages

1. Create `.rst` files in the appropriate directory
2. Add the file to the relevant `toctree` directive
3. Use proper reStructuredText formatting

### API Documentation

API documentation is auto-generated from docstrings. To add new functions:

1. Write comprehensive docstrings in your Python code
2. Add the function to the appropriate `api/*.rst` file
3. Use `.. autofunction::` directive

### Mathematical Notation

Use MathJax for mathematical expressions:

```rst
Inline math: :math:`x = y + z`

Display math:

.. math::

   \frac{\partial L}{\partial \theta} = \sum_{i=1}^n \nabla_\theta f(x_i, \theta)
```

### Code Examples

Use code blocks with syntax highlighting:

```rst
.. code-block:: python

   import torchsparsegradutils as tsgu
   result = tsgu.sparse_mm(A, B)
```

### Cross-References

Link to other parts of the documentation:

```rst
See :doc:`quickstart` for basic usage.
See :func:`torchsparsegradutils.sparse_mm` for details.
```

## Troubleshooting

### Common Issues

**Build fails with locale error:**
```bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
make html
```

**Missing dependencies:**
```bash
uv sync --locked --group docs
```

**Warnings about missing references:**
- Check that all referenced files exist
- Verify function names in `.. autofunction::` directives
- Ensure proper indentation in `.rst` files

### Getting Help

- Check the [Sphinx documentation](https://www.sphinx-doc.org/)
- See [reStructuredText guide](https://docutils.sourceforge.io/rst.html)
- Review existing documentation files for examples
