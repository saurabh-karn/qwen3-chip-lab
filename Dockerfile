FROM python:3.12-slim

WORKDIR /app
COPY services/chip_lab/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY web /app/web
COPY services /app/services
COPY lab /app/lab
COPY qwen_ref /app/qwen_ref
COPY configs /app/configs
COPY evidence /app/evidence
COPY spec /app/spec

ENV PYTHONPATH=/app
ENV HOST=0.0.0.0
ENV PORT=8080
ENV LAB_COOKIE_SECURE=1
EXPOSE 8080

CMD ["python", "-m", "services.chip_lab"]
