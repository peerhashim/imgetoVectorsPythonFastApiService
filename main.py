from fastapi import FastAPI
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import logging

app = FastAPI(title="Face Embedding Service")

# Logging
logging.basicConfig(level=logging.INFO)

# Lazy-loaded model
face_model = None


def get_model():
    global face_model

    if face_model is None:
        from insightface.app import FaceAnalysis

        face_model = FaceAnalysis(
            name="buffalo_s",
            providers=["CPUExecutionProvider"]
        )

        # Use CPU mode
        face_model.prepare(ctx_id=-1, det_size=(640, 640))

        logging.info("Face model loaded successfully")

    return face_model


class ImageRequest(BaseModel):
    base64: str


class FaceResponse(BaseModel):
    vector: list[float]


@app.get("/")
def health_check():
    return {
        "status": "running",
        "service": "Face Embedding API"
    }


@app.post("/embed", response_model=FaceResponse)
def embed_face(req: ImageRequest):
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(req.base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            logging.error("Invalid image received")
            return {"vector": []}

        model = get_model()
        faces = model.get(img)

        if not faces:
            logging.info("No face detected")
            return {"vector": []}

        embedding = faces[0].embedding.tolist()

        return {"vector": embedding}

    except Exception as e:
        logging.exception("Error while generating embedding")
        return {"vector": []}