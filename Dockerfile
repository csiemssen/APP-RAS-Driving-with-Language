## BUILD FLASH-ATTN
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu20.04 AS dependencies

ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"

RUN apt-get update && \
  apt-get install -y python3-pip python3-dev python-is-python3 python3.12-venv && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set up a venv so we can install deps
RUN python3 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
  pip install --no-cache-dir -r /tmp/requirements.txt && \
  pip install --no-cache-dir flash-attn --no-build-isolation


## FINAL BUILD
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04 AS final

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/venv/bin:$PATH"

RUN apt-get update && \
  apt-get install -y python3-pip python3-dev python-is-python3 python3-venv && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=dependencies /app/venv /app/venv

# Copy project files
COPY . /app

CMD ["python", "main.py"]
