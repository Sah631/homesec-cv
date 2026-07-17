# Home Security Computer Vision System

A local multi-camera computer vision system for real-time CCTV object detection, annotated RTSP streaming, and event-based clip recording.

The system reads RTSP camera streams, performs batched object detection, saves clips when relevant objects are detected, and re-streams a live annotated 2x2 camera grid over RTSP for viewing on local devices.

The goal is to explore practical computer vision system design beyond single-image inference, including real-time streams, multi-camera coordination, event recording, and edge deployment.

## Current Status

This repository currently contains a Python/OpenCV-based implementation tested on a desktop PC with:

- 32 GB RAM
- 12-core CPU
- NVIDIA RTX 2080 8 GB GPU

A future version is planned for deployment on a Jetson Orin Nano using NVIDIA DeepStream and TensorRT.

## Features

- Multi-camera RTSP ingestion with threaded camera, inference, streaming, and clip-saving workers
- Batched object detection across camera feeds
- Annotated RTSP grid streaming using FFmpeg and MediaMTX
- Event-based clip recording with pre-roll, post-roll, and maximum duration limits
- Configurable detection classes, camera streams, and local-only deployment settings

## System Overview

The current pipeline is:

```text
CCTV RTSP Streams
        ↓
Camera Worker Threads
        ↓
Frame Queues
        ↓
Batched Object Detection
        ↓
Detection Packets
        ↓
Annotated Frames
        ↓
        ├── Live RTSP Grid Stream
        └── Event-Based Clip Saving
```

## Clip Recording Logic

When a target object is detected, the system saves an event clip with 10 seconds of pre-roll, 5 seconds of post-roll, and a maximum duration of 1 minute. This captures context around the event while preventing long-running clips.

## Live RTSP Streaming

The annotated 2x2 camera grid is encoded with FFmpeg and published through MediaMTX as a local RTSP stream:

```text
rtsp://<device-ip>:8554/annotated_grid
```

For development, the stream is intended to run only on the local network.

## Configuration

Runtime configuration is handled through environment variables and `config.py`.

Typical configuration includes:

```env
CAMERA_1_RTSP_URL=rtsp://...

MEDIAMTX_HOST=127.0.0.1
MEDIAMTX_PORT=8554
MEDIAMTX_PATH=annotated_grid
MEDIAMTX_PUBLISH_USERNAME=publisher
MEDIAMTX_PUBLISH_PASSWORD=change_me
```

## Running the System

### 1. Start MediaMTX

Run MediaMTX separately:

```bash
./mediamtx mediamtx.yml
```

### 2. Start the Python Application

```bash
python main.py
```

### 3. View the RTSP Stream

Open VLC and connect to:

```text
rtsp://<device-ip>:8554/annotated_grid
```

If authentication is enabled:

```text
rtsp://<username>:<password>@<device-ip>:8554/annotated_grid
```

## Security Notes

This system is intended for local network use only. Do not expose the RTSP port to the internet or configure router port forwarding. Use MediaMTX authentication, restrict publishing to localhost, and keep camera/RTSP credentials out of GitHub.

## Roadmap

Planned improvements include Jetson Orin Nano deployment, a DeepStream/TensorRT backend, hardware-accelerated video decode/encode, higher-resolution event recording, ignore zones, improved object persistence, and a web dashboard for event browsing.
