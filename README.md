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

Open the gear menu and choose **ตั้งค่าเสียงแจ้งเตือน** to pick which alert
channels are active. The looping siren is DANGER-only; Thai voice announces
both DANGER and WARNING with the object type and distance, while SAFE is silent.
The voice also waits for a stable detection and skips repeating the same object
unless its distance has moved at least 3 m, reducing false repeats from detector
flicker.

The runtime selects an installed Windows Speech voice whose culture is Thai
(`th-*`) and never silently falls back to an English voice. If no Thai voice
is installed, add Thai speech in Windows Language settings. Legacy MP3 clips
are not used when a measured distance is available, so the app cannot silently
announce an incomplete or English distance. Distances are converted to Thai
words (for example, 15.8 becomes “สิบห้า เมตร” in the voice). The bundled
voice is intentionally slowed down and pauses around the distance for clarity.

For deployment to other computers, the project includes an offline Thai VachanaTTS
male Thai ONNX voice (`th_m_1`) in `tts/voices/` as the default. Install the Python dependencies from
`requirements.txt`; the application uses this bundled voice before checking
other TTS backends or Windows Speech.

## Notes

## วิธีใช้งานแบบควบคุมได้

### เข้า environment `cvce`

เปิด PowerShell ในโฟลเดอร์โปรเจกต์:

```powershell
cd C:\CS3\project\pg
.\cvce\Scripts\Activate.ps1
```

ถ้า PowerShell ปิดกั้นสคริปต์ ให้ใช้เฉพาะหน้าต่างนี้:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\cvce\Scripts\Activate.ps1
```

ตรวจสอบ environment:

```powershell
where.exe python
python --version
```

ผลลัพธ์ควรชี้ไปที่ `cvce\Scripts\python.exe` จากนั้นติดตั้งและเปิดโปรแกรม:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

### การควบคุมในหน้าหลัก

| ปุ่ม | หน้าที่ |
|---|---|
| เปิดกล้อง | เริ่มตรวจจับจากกล้อง |
| ปิดกล้อง | หยุดกล้องและหยุดการตรวจจับ |
| เปิดวิดีโอ | เลือกไฟล์วิดีโอเพื่อตรวจจับ |
| ทดสอบรูปภาพ | ตรวจจับภาพหนึ่งภาพหรือหลายภาพ |
| บันทึกผล | บันทึกภาพผลลัพธ์และ CSV |
| เสียงแจ้งเตือน | ปิดหรือเปิดไซเรนและเสียงพูด |
| ตั้งค่า | ตั้งค่าระยะ โซน กล้อง โมเดล และเสียง |

### หลักการแจ้งเตือน

- ระยะอันตราย: พูด `อันตะราย มี [วัตถุ] อยู่ในระยะ [จำนวนเต็ม] เมตร`
- ระยะระวัง: พูด `ระวัง มี [วัตถุ] อยู่ในระยะ [จำนวนเต็ม] เมตร`
- ระยะปลอดภัย: ไม่พูดแจ้งเตือน
- วัตถุอันตรายมีลำดับความสำคัญสูงกว่าวัตถุระวัง
- ระบบลดการพูดซ้ำและรอผลตรวจจับให้คงที่ก่อนแจ้งเตือน

## เครดิตโมเดลเสียงภาษาไทย

ไฟล์เสียงภาษาไทยใน `tts/voices/` ใช้โมเดล **VachanaTTS** แบบ ONNX จาก:

- [VachanaTTS บน Hugging Face](https://huggingface.co/VIZINTZOR/VachanaTTS)
- [VachanaTTS source repository](https://github.com/VYNCX/VachanaTTS)
- [PyThaiTTS](https://github.com/PyThaiNLP/PyThaiTTS)

โปรดตรวจสอบ license จากแหล่งต้นทางก่อนนำโปรแกรมหรือโมเดลไปแจกจ่ายเชิงพาณิชย์

## Performance and deployment notes

- YOLO model loading is performed by its worker thread so the interface can
  remain responsive while the model initializes.
- Detection distance is an estimate based on bounding-box height, calibrated
  focal length, and approximate object height. Calibrate the camera before
  relying on the distance for safety decisions.
- Thai TTS models are local ONNX files. Only the selected voice model should
  be loaded; switching voices may briefly use CPU while the new model loads in
  the background.
- Run the basic logic checks with `python -m unittest discover tests`.

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
