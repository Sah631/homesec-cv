# Entrypoint for app
# Should initialise model and start video processing loop
from models.yolo26 import model
from video import process_videos


def main():
    # Have separate functions for loading model and starting the video processing loop

    yolo_model = model
    process_videos(yolo_model)


if __name__ == "__main__":
    main()
