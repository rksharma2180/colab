# Technical Video Transcriber (v3.0 Modular & Drive Integrated)

High-accuracy transcription for technical courses using faster-whisper with OpenAI `large-v3` model.

## Features
- 🎙️ GPU-accelerated transcription (faster-whisper + large-v3)
- ⚙️ **Modular Pre-Processing (`media_processor.py`)**: Smart Bitrate Analyzer prevents size inflation
- ⚡ **Audio Extraction Mode**: Converts videos to 64k mono MP3 before loading AI model (saves 95% space)
- 📁 **Google Drive Backup**: Saves compressed media & transcripts directly to Drive
- 🧹 **Automatic Drive Cleanup**: Automatically deletes heavy original videos from Drive once transcribed
- ⏩ **Resume/Disconnect-Resistant**: Skips already completed transcripts if Colab disconnects
- 🔧 **Tech Speech-to-Symbol**: `slash`→`/`, `dot com`→`.com`, `asterisk`→`*`, `dash dash`→`--`
- 🔢 **Number Verbalization**: `eight zero eight zero`→`8080`, `port twenty six`→`port 26`
- 📝 **Proper Nouns & Java Support**: 200+ terms (`ArrayList`, `NullPointerException`, `JVM`, `Docker`, `Spring Boot`)
- 📄 **TurboScribe-style Paragraphs**: Formats output in ~45-word readable blocks

## Google Colab Execution (Single Cell)

```python
# 1. Mount Google Drive (if not already mounted from sidebar)
from google.colab import drive
drive.mount('/content/drive')

# 2. Download modular scripts
!wget -q https://raw.githubusercontent.com/rksharma2180/colab-transcriber/main/media_processor.py -O media_processor.py
!wget -q https://raw.githubusercontent.com/rksharma2180/colab-transcriber/main/transcriber.py -O transcriber.py

# 3. Run transcriber (Options: video, audio, or none)
%run transcriber.py video
```

## Folder Structure in Google Drive (`/MyDrive/Colab_Transcriber/`)
- `original/`: Upload your video files here (auto-deleted after transcription to save space).
- `compressed/`: Contains compressed `.mp4` or `.mp3` audio files.
- `transcripts/`: Stores final `.txt` transcripts permanently.

## License
MIT
