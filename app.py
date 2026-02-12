from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from insightface.app import FaceAnalysis

app = FastAPI(title="Face Embedding Service")

# Load ArcFace model (buffalo_l)
model = FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 GPU, -1 CPU

class ImageRequest(BaseModel):
    base64: str

class FaceResponse(BaseModel):
    vector: list[float]

@app.post("/embed", response_model=FaceResponse)
def embed_face(req: ImageRequest):
    # Decode base64
    img_bytes = base64.b64decode(req.base64)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    faces = model.get(img)

    if len(faces) == 0:
        return {"vector": []}

    embedding = faces[0].embedding.tolist()  # 512D vector
    return {"vector": embedding}
