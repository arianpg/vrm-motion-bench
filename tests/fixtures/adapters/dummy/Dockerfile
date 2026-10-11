# ビルドコンテキストはリポジトリのルート:
#   docker build -f tests/fixtures/adapters/dummy/Dockerfile -t vmb-adapter-dummy .
FROM python:3.11-slim
COPY packages/sourcemotion /opt/vmb/packages/sourcemotion
COPY packages/adapter_kit /opt/vmb/packages/adapter_kit
RUN pip install --no-cache-dir /opt/vmb/packages/sourcemotion \
 && pip install --no-cache-dir --no-deps /opt/vmb/packages/adapter_kit \
 && pip install --no-cache-dir "fastapi>=0.115" "uvicorn>=0.30"
COPY tests/fixtures/adapters/dummy/server /opt/vmb/adapter
WORKDIR /opt/vmb/adapter
CMD ["python", "dummy_adapter.py"]
