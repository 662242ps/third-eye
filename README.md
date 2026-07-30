# Third Eye

Desktop application for detecting road objects from a camera or video, with configurable warning distance and warning zone.

## Setup

Use Python 3.10 or newer, then create a fresh virtual environment:

```powershell
py -3.10 -m venv cvce
.\cvce\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

`models/best.pt` is required and is included in this repository.

## Distance warning setting

Open the gear menu and choose **Distance Setting** to adjust the distance thresholds in meters.

Default values:

- Danger: distance <= 5 m
- Warning: distance <= 20 m
- Safe: distance > 20 m

The values are saved in `settings/distance_thresholds.json` and are reloaded during detection.

## Notes

- Configure the polygon in **Zone Setting**. The active zone is saved in `zones/active.txt`.
- Camera index and focal-length calibration are available from the gear menu
  under **Camera Setting** and saved in `settings/camera_settings.json`.
- Use **Test หลายภาพ** to select one or more still images. Test mode processes
  every image across the full frame without applying the active zone, while
  retaining estimated distance and risk labels.
- After a batch finishes, use **บันทึกผล** to export annotated JPEG images and
  a UTF-8 `detection_results.csv` summary.
- Browse processed images with the previous/next buttons or the mouse wheel
  while the pointer is over the image.
- Uploaded videos show a timeline with current/duration time. Drag the slider
  to seek to another point in the clip.
- Reported distances are estimates. Calibrate `FOCAL_LENGTH` in `vision/yolo_thread.py` against the camera used in deployment.
