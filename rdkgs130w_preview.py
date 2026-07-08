#!/usr/bin/env python3
import argparse
import signal
import sys
import time

from hobot_vio import libsrcampy

running = True


def _stop(signum, frame):
    global running
    running = False


def main():
    parser = argparse.ArgumentParser(description="Live HDMI/X11 preview for rdkGS130W MIPI camera")
    parser.add_argument("-w", "--width", type=int, default=1088)
    parser.add_argument("-H", "--height", type=int, default=1280)
    parser.add_argument("-f", "--fps", type=int, default=30)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    disp = libsrcampy.Display()
    cam = libsrcampy.Camera()
    opened_display = False
    opened_camera = False

    try:
        disp.display(0, args.width, args.height)
        opened_display = True

        ret = cam.open_cam(0, -1, args.fps, args.width, args.height)
        if ret:
            print("Error: Failed to open camera.", file=sys.stderr)
            return 1
        opened_camera = True

        ret = libsrcampy.bind(cam, disp)
        if ret:
            print(f"Error: bind camera to display failed: {ret}", file=sys.stderr)
            return 1

        print(f"Preview running: {args.width}x{args.height}@{args.fps}. Press Ctrl+C to stop.", flush=True)
        while running:
            time.sleep(0.2)
    finally:
        try:
            if opened_camera and opened_display:
                libsrcampy.unbind(cam, disp)
        except Exception as exc:
            print(f"Warning: unbind failed: {exc}", file=sys.stderr)
        try:
            if opened_display:
                disp.close()
        except Exception as exc:
            print(f"Warning: display close failed: {exc}", file=sys.stderr)
        try:
            if opened_camera:
                cam.close_cam()
        except Exception as exc:
            print(f"Warning: camera close failed: {exc}", file=sys.stderr)

    print("Preview stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
