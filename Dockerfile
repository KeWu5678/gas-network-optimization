# Reproducibility image: locked dependencies (uv.lock) + the package.
# Default command runs the test suite; the examples run against a mounted
# results/ (and data/ for the GasLib instances, see README):
#
#   docker build -t gasnetopt .
#   docker run --rm gasnetopt
#   docker run --rm -v "$PWD/results:/app/results" gasnetopt \
#       uv run python examples/run_translines.py
FROM python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf

COPY --from=ghcr.io/astral-sh/uv:0.11@sha256:0f36cb9361a3346885ca3677e3767016687b5a170c1a6b88465ec14aefec90aa /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

CMD ["uv", "run", "pytest", "-q"]
