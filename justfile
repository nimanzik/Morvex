default:
    @just --list

clean:
    @find . -type d -name "__pycache__" -exec rm -rf {} +
    @find . -type d -name ".pytest_cache" -exec rm -rf {} +
    @find . -type d -name ".ruff_cache" -exec rm -rf {} +
    @rm -rf dist/

uv_quality_options := "--frozen --isolated --no-dev --group quality"
uv_test_options := "--frozen --isolated --no-dev --group test --extra torch-cpu"
pytest_options := "-v --tb=short"
default_python_version := "3.13"

lint:
    @uv run {{ uv_quality_options }} ruff check --fix

lint-check:
    @uv run {{ uv_quality_options }} ruff check

format:
    @uv run {{ uv_quality_options }} ruff format

format-check:
    @uv run {{ uv_quality_options }} ruff format --check

typecheck:
    @uv run {{ uv_quality_options }} --group test --extra torch-cpu ty check

test python_version=default_python_version:
    @uv run {{ uv_test_options }} --python "{{ python_version }}" \
        pytest -rs {{ pytest_options }} tests/

# Run the checks used by CI on the default Python version.
ci: lint-check format-check typecheck test build-check

build-check python_version=default_python_version:
    @rm -rf dist/
    uv build --python "{{ python_version }}"
    uv run --frozen --isolated --no-dev --group build --python "{{ python_version }}" \
        twine check dist/*
