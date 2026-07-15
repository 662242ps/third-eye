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

Click **Distance Setting** in the main window to adjust the distance thresholds in meters.

Default values:

- Danger: distance <= 5 m
- Warning: distance <= 20 m
- Safe: distance > 20 m

The values are saved in `settings/distance_thresholds.json` and are reloaded during detection.

## Notes

- Configure the polygon in **Zone Setting**. The active zone is saved in `zones/active.txt`.
- Reported distances are estimates. Calibrate `FOCAL_LENGTH` in `vision/yolo_thread.py` against the camera used in deployment.
