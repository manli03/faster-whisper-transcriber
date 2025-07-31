# Faster-Whisper Transcriber

A simple command-line tool to transcribe audio files using the `faster-whisper` library. It supports multiple output formats (SRT, JSON, TXT) and can be run with command-line arguments or through an interactive setup.

## Installation

1.  Make sure you have Python 3.10 or newer installed.
2.  Navigate to the `Python` directory where this file is located.
3.  Install the script and its dependencies:
    ```sh
    pip install .
    ```

## Usage

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