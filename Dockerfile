FROM python:3.12-slim

WORKDIR /app

# Optional dependency groups: anthropic (Claude provider), mcp (stdio MCP
# server), agent (Claude Agent SDK backend). Override with
#   docker build --build-arg EXTRAS=anthropic,mcp,agent .
ARG EXTRAS=anthropic,mcp

COPY pyproject.toml README.md LICENSE ./
COPY watchwise ./watchwise
RUN pip install --no-cache-dir ".[${EXTRAS}]"

ENV PYTHONUNBUFFERED=1 \
    WATCHWISE_DB_PATH=/data/watchwise.db

VOLUME ["/data"]
EXPOSE 5055

CMD ["watchwise", "serve", "--host", "0.0.0.0", "--port", "5055"]
