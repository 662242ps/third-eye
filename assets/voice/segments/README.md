# Thai alert voice segments

Generate the complete offline word set once from the bundled voice model:

```bat
cd C:\CS3\project\pg
cvce\Scripts\activate.bat
python tools\generate_voice_segments.py
```

The generator creates the danger/warning words, all supported object names,
the connector words, `เมตร`, and numbers `0` through `99` for the bundled
`th_m_1.onnx` voice. The live video path concatenates these WAV files and does
not load ONNX/TTS.
