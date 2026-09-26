# RetiLink - Hugging Face Docker Space
FROM python:3.10-slim
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY --chown=user deploy/space/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user retilink retilink
COPY --chown=user scripts scripts
COPY --chown=user weights weights
USER user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface \
    RETILINK_APP_ROOT=/tmp/retilink RETILINK_MODEL_DIR=/home/user/app/weights PYTHONUNBUFFERED=1
# pre-fetch the DINOv2-L base weights at build time so the first analysis is not a 1.2 GB download
RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-large')"
EXPOSE 7860
# synthetic workspace (no mBRSET images), then serve
CMD python scripts/seed_demo.py --reset --no-images && uvicorn retilink.app.main:app --host 0.0.0.0 --port 7860
