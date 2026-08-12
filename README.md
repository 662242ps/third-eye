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

## Model setting

Open the gear menu and choose **ตั้งค่าโมเดล** to pick which YOLO weights
file to run. Any `*.pt` file placed anywhere under `models/` is selectable
from the dropdown — it no longer has to be named `best.pt`. Use **เพิ่มโมเดล
จากไฟล์ในเครื่อง...** to import a `.pt` file from elsewhere on disk (it is
copied into `models/`). The selection is saved in
`settings/model_settings.json`; switching models restarts the detection
thread with the new weights.

## Alert sound setting

Open the gear menu and choose **ตั้งค่าเสียงแจ้งเตือน** to pick which DANGER
alert channels are active: the looping siren, the Thai voice announcement,
both, or neither. The choice is saved in `settings/alert_settings.json` and
applies on top of the toolbar's 🔊/🔇 mute button. The voice announcement
also skips repeating for the same object unless its distance has moved at
least 3 m since it was last announced, so a vehicle parked at a steady
range doesn't get re-announced every few seconds.

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
- Bounding boxes are velocity-predicted between inference results (see `YoloThread._predict_box`) so fast-moving objects stay tracked and the drawn box doesn't lag behind, even though inference runs slower than the display refresh rate.

## Danger alerts (sound + voice)

When a tracked object enters **DANGER** range during live camera/video playback:

- A looping siren plays (`assets/alert_danger.wav`) via `winsound` — see `vision/alert_sound.py`.
- A Thai voice clip names the nearest dangerous object (e.g. "ระวัง มีรถยนต์อยู่ใกล้"), throttled to once every 3 seconds per label — see `vision/voice_alert.py`. Clips live in `assets/voice/*.mp3` and are played through the Windows MCI API (`winmm.dll`), so no ffmpeg or audio library is required at runtime.
- Both are Windows-only (they no-op elsewhere) and can be muted from the **🔊 เสียงเตือน** button in the toolbar.
- Neither plays during **Test หลายภาพ** batch testing, only for live camera/video frames.

To add or re-record a voice phrase (e.g. after adding a new detectable class), regenerate the MP3 with [gTTS](https://pypi.org/project/gTTS/) (`pip install gTTS`, dev-time only):

```python
from gtts import gTTS
gTTS(text="ระวัง มีรถบัสอยู่ใกล้", lang="th").save("assets/voice/bus_danger.mp3")
```
