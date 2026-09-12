ARG PYTHON_BASE_IMAGE=python:3.13-slim-bookworm
FROM ${PYTHON_BASE_IMAGE}

ARG RUNTIME_UID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JIEBA_CACHE_DIR=/tmp/jieba-cache

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --requirement requirements.txt

COPY . .
COPY docker/jieba-cache /app/.jieba_cache

USER root
RUN chmod 1777 /app/.jieba_cache
USER ${RUNTIME_UID}

ENTRYPOINT ["python"]
CMD ["run_evaluation.py", "--help"]
