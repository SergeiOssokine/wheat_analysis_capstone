FROM python:3.12-slim
RUN pip install uv
COPY ["pyproject.toml","uv.lock","./"]
RUN uv sync --frozen --no-dev --no-install-project
COPY ["grain_prediction_service.py","data_loader.py","resnet50.onnx","resnet50.onnx.data","./"]
CMD ["uv", "run", "uvicorn", "grain_prediction_service:app", "--host", "0.0.0.0", "--port", "9696"]
EXPOSE 9696