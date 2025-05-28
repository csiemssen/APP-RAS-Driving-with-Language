# Use NVIDIA's PyTorch image with CUDA support
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
FROM nvcr.io/nvidia/pytorch:25.04-py3

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
  unzip && \
  apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# Create working directory
WORKDIR /app

# Copy project files
COPY . /app

# Pull the models during build
RUN python -c "from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration; \
  AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct'); \
  Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct')"

# Pull the dataset
RUN mkdir -p /app/data/drivelm
RUN gdown --id 1CvTPwChKvfnvrZ1Wr0ZNVqtibkkNeGgt --output /app/data/drivelm/v1_1_train_nus.json
RUN gdown --id 1fsVP7jOpvChcpoXVdypaZ4HREX1gA7As --output /app/data/drivelm/v1_1_val_nus_q_only.json
RUN gdown --id 1DeosPGYeM2gXSChjMODGsQChZyYDmaUz --output /app/data/drivelm_nus_imgs_train.zip && \
  unzip /app/data/drivelm_nus_imgs_train.zip -d /app/data && \
  rm /app/data/drivelm_nus_imgs_train.zip
RUN gdown --id 18f8ygNxGZWat-crUjroYuQbd39Sk9xCo --output /app/data/drivelm_nus_imgs_val.zip && \
  unzip /app/data/drivelm_nus_imgs_val.zip -d /app/data && \
  rm /app/data/drivelm_nus_imgs_val.zip
RUN cp -r /app/data/val_data/* /app/data/nuscenes/samples/ && \
  rm -rf /app/data/val_data
