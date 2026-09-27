# Insight Rx vision worker on a Hugging Face Space (free CPU basic: 2 vCPU, 16 GB RAM). No PyTorch, no GPU.
FROM python:3.11-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user . .
EXPOSE 7860
CMD ["bash", "start.sh"]
