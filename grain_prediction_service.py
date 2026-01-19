import pickle
from typing import Any

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from PIL import Image
import onnxruntime as ort
from data_loader import get_transforms, IMGNET_MEANS, IMGNET_STDS
import io

# Note that we find the pipeline file!
onnx_path = "resnet10_final.onnx"
from fastapi import File, UploadFile, HTTPException


app = FastAPI(title="predict_api_service")


@app.post("/predict")
async def predict(input: UploadFile = File(...)):
    contents = await input.read()
    ort_session = ort.InferenceSession(onnx_path)
    image = Image.open(io.BytesIO(contents))
    # Get and apply image transforms
    img_transforms = get_transforms("inference", means=IMGNET_MEANS, stds=IMGNET_STDS)
    tmp = img_transforms(image)

    # Change dimensions from (3,224,224) to (1,3,224,224)
    input_tensor = tmp.unsqueeze(0)
    # Convert to numpy for ONNX
    input_numpy = input_tensor.numpy()
    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: input_numpy}
    ort_outputs = ort_session.run(None, ort_inputs)
    # Get prediction
    output = ort_outputs[0]
    predicted_class = int(np.argmax(output, axis=1)[0])
    
    result = {"predicted_class": predicted_class}
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
