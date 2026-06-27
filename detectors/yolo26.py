from ultralytics import YOLO

MODEL_PATH = "checkpoints/yolo26m.pt"


class YOLODetector:
    def __init__(self, model_path: str = MODEL_PATH, model_name: str = "YOLO26"):
        self.model = YOLO(model_path)
        self.model_name = model_name

    def predict(self, frames, **kwargs):
        return self.model.predict(frames, **kwargs)

    def get_model_name(self) -> str:
        return self.model_name


model = YOLO("checkpoints/yolo26m.pt")
