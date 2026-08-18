# Colab Transcriber

High-accuracy transcription for technical courses using faster-whisper with OpenAI large-v3 model.

## Features
- 🎙️ GPU-accelerated transcription (faster-whisper + large-v3)
- 🔧 Tech speech-to-symbol conversion (`slash`→`/`, `dot com`→`.com`, `asterisk`→`*`)
- 🔢 Number verbalization (`eight zero eight zero`→`8080`, `port twenty six`→`port 26`)
- 📝 Proper noun capitalization (Docker, Kubernetes, Spring Boot, etc.)
- 📄 TurboScribe-style paragraph formatting
- 🛡️ Anti-hallucination measures
- 📦 Multi-file batch processing with auto-download

## Usage in Google Colab

### Option 1: Pull from GitHub (Recommended)
```python
!wget https://raw.githubusercontent.com/YOUR_USERNAME/colab-transcriber/main/transcriber.py
%run transcriber.py
```

### Option 2: Copy-Paste
Copy the contents of `transcriber.py` into a Colab cell and run it.

## How It Works
1. Upload video/audio files when prompted
2. Script auto-installs `faster-whisper` and detects GPU
3. Transcribes each file with proper noun + tech symbol processing
4. Downloads results as `.txt` files (or `.zip` for multiple)

## Supported Formats
`.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`

## Tech Speech Patterns

| Speaker Says | Output |
|-------------|--------|
| "slash" | `/` |
| "dot com" | `.com` |
| "asterisk" | `*` |
| "dash dash version" | `--version` |
| "localhost colon eight zero eight zero" | `localhost:8080` |
| "javac space dash version" | `javac -version` |
| "ls dash la asterisk dot java" | `ls -la *.java` |
| "www dot google dot com" | `www.google.com` |
| "open curly brace" | `{` |

## License
MIT
