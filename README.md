# Faster-Whisper Transcriber  

## 📖 Description  
A simple **command-line tool** to transcribe audio files using the `faster-whisper` library.  
It supports multiple output formats (**SRT, JSON, TXT**) and can be used either by passing **command-line arguments** or through an **interactive setup**.  

## 🔑 Key Features  
- 🎧 **Audio Transcription** – Converts speech into text with high accuracy.  
- 🌍 **Language Support** – Specify transcription language for better results.  
- 📂 **Multiple Output Formats** – Export transcripts as **SRT, JSON, or TXT**.  
- ⚡ **Flexible Usage** – Run with **arguments** for automation or use the **interactive mode** for guided setup.  
- 🛠️ **Customizable Models** – Choose from different Whisper model sizes (e.g., `large-v3`).  

---

## 📦 Installation  

1. Make sure you have **Python 3.10 or newer** installed.  
2. Navigate to the `Python` directory where this file is located.  
3. Install the script and its dependencies:  
   ```sh
   pip install .

## ▶️ Usage

After installation, you can run the tool from anywhere in your terminal.

### Command-Line Mode

Provide the audio file and other options as arguments:

```sh
transcribe --file "path/to/your/audio.mp3" --model_size large-v3 --language en --formats srt,txt
```

### Interactive Mode

If you run the command without any arguments, it will launch an interactive setup to guide you through the process:

```sh
transcribe
```

## 💡 Use Cases

- 🎙️ Podcasts & Interviews – Quickly generate transcripts for editing or publishing.
- 🏫 Lectures & Meetings – Capture spoken content for notes or study materials.
- 📺 Subtitles – Export to .srt for easy subtitle integration in videos.

---

This tool makes **speech-to-text transcription simple, flexible, and efficient**, whether for personal use, research, or media production.
