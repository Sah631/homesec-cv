from ultralytics import YOLO

from config import MODEL_PATH


# TODO: Add an abstract class with predict and get_model_name methods, and make these specific for each model in their respective classes
class YOLODetector:
    def __init__(self, model_path: str = MODEL_PATH, model_name: str = "YOLO26"):
        self.model = YOLO(model_path)
        self.model_name = model_name

    def predict(self, frames, **kwargs):
        return self.model.predict(frames, **kwargs)

    def get_model_name(self) -> str:
        return self.model_name
