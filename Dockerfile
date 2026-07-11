# Reproducibility image: locked dependencies (uv.lock) + the package.
# Default command runs the test suite; the examples run against a mounted
# results/ (and data/ for the GasLib instances, see README):
#
#   docker build -t gasnetopt .
#   docker run --rm gasnetopt
#   docker run --rm -v "$PWD/results:/app/results" gasnetopt \
#       uv run python examples/run_translines.py
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

CMD ["uv", "run", "pytest", "-q"]
