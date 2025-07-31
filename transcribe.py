
import argparse
import os
import json
import torch
from faster_whisper import WhisperModel
from datetime import timedelta
from colorama import Fore, Style, init
import time
from tqdm import tqdm


# Initialize colorama
init(autoreset=True)

def format_srt_timestamp(seconds):
    """Formats seconds into HH:MM:SS,ms for SRT files."""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def write_srt(segments, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_timestamp(seg.start)} --> {format_srt_timestamp(seg.end)}\n")
            f.write(f"{seg.text.strip()}\n\n")

def write_txt(segments, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"{seg.text.strip()}\n")

def write_json(segments, out_path):
    output = []
    for seg in segments:
        output.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

def get_user_choice(prompt_text, options, default=None):
    """Prompts user to select from a list of options."""
    print(prompt_text)
    for i, option in enumerate(options, 1):
        if option == default:
            print(f"  {i}. {option} [default]")
        else:
            print(f"  {i}. {option}")

    while True:
        try:
            choice_str = input(f"Enter your choice (number): ").strip()
            if not choice_str and default is not None:
                return default
            choice = int(choice_str)
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                print(f"{Fore.RED}Invalid choice. Please select a number between 1 and {len(options)}.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Invalid input. Please enter a number.{Style.RESET_ALL}")

def interactive_setup(args):
    """Prompts user for settings if not provided via command line."""
    print(f"\n{Fore.CYAN}--- Interactive Setup ---{Style.RESET_ALL}")
    print("No file provided via arguments, entering interactive mode.")

    # File Path
    while True:
        file_path = input(f"\n{Fore.YELLOW}📂 Enter the path to your audio file: {Style.RESET_ALL}").strip()
        # Remove quotes if the user pasted a path wrapped in them (e.g., from "Copy as path" in Windows)
        file_path = file_path.strip('"').strip("'")
        if os.path.exists(file_path) and os.path.isfile(file_path):
            args.file = file_path
            break
        else:
            print(f"{Fore.RED}Error: File not found or is not a valid file. Please try again.{Style.RESET_ALL}")

    # Device
    has_gpu = torch.cuda.is_available()
    available_devices = ["cuda", "cpu"] if has_gpu else ["cpu"]
    default_device = "cuda" if has_gpu else "cpu"
    if len(available_devices) > 1:
        args.device = get_user_choice(f"\n{Fore.YELLOW}💻 Choose a device to run on:{Style.RESET_ALL}", available_devices, default=default_device)
    else:
        args.device = default_device
        print(f"\n{Fore.YELLOW}💻 Using device: {args.device}{Style.RESET_ALL}")

    # Model Size
    model_sizes = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
    args.model_size = get_user_choice(f"\n{Fore.YELLOW}🧠 Choose a model size (smaller is faster):{Style.RESET_ALL}", model_sizes, default="medium")

    # Compute Type
    if args.device == "cuda":
        # float16 is recommended for NVIDIA GPUs
        compute_types = ["float16", "int8_float16", "int8"]
        default_compute = "float16"
        prompt_text = f"\n{Fore.YELLOW}⚙️  Choose a compute type ('float16' is recommended for GPU):{Style.RESET_ALL}"
    else: # cpu
        compute_types = ["int8", "float32"]
        default_compute = "int8"
        prompt_text = f"\n{Fore.YELLOW}⚙️  Choose a compute type ('int8' is recommended for CPU):{Style.RESET_ALL}"
    args.compute_type = get_user_choice(prompt_text, compute_types, default=default_compute)

    # CPU Threads (only if using CPU)
    if args.device == "cpu":
        while True:
            try:
                cpu_threads_str = input(f"\n{Fore.YELLOW}🧵 Enter number of CPU threads (0 for auto) [default: 0]: {Style.RESET_ALL}").strip()
                if not cpu_threads_str:
                    args.cpu_threads = 0
                    break
                cpu_threads = int(cpu_threads_str)
                if cpu_threads >= 0:
                    args.cpu_threads = cpu_threads
                    break
                else:
                    print(f"{Fore.RED}CPU threads must be a non-negative integer.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Invalid input. Please enter an integer.{Style.RESET_ALL}")

    # Beam Size
    while True:
        try:
            beam_size_str = input(f"\n{Fore.YELLOW}🎯 Enter beam size (e.g., 1 for speed, 5 for accuracy) [default: 5]: {Style.RESET_ALL}").strip()
            if not beam_size_str:
                args.beam_size = 5
                break
            beam_size = int(beam_size_str)
            if beam_size > 0:
                args.beam_size = beam_size
                break
            else:
                print(f"{Fore.RED}Beam size must be a positive integer.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Invalid input. Please enter an integer.{Style.RESET_ALL}")

    # Language
    lang_input = input(f"\n{Fore.YELLOW}🌍 Enter language code (e.g., en, ms) or leave blank for auto-detect: {Style.RESET_ALL}").strip()
    args.language = lang_input if lang_input else None

    # Output Directory
    output_dir_input = input(f"\n{Fore.YELLOW}📁 Enter output directory [default: output]: {Style.RESET_ALL}").strip()
    args.output_dir = output_dir_input if output_dir_input else "output"

    # Formats
    formats_input = input(f"\n{Fore.YELLOW}💾 Enter output formats (srt,json,txt) separated by commas [default: srt]: {Style.RESET_ALL}").strip().lower()
    args.formats = formats_input if formats_input else "srt"

    print(f"\n{Fore.CYAN}--- Setup Complete. Starting transcription... ---\n{Style.RESET_ALL}")
    return args

def main():
    main_start_time = time.time()

    parser = argparse.ArgumentParser(description="🎧 Transcribe audio using faster-whisper")
    parser.add_argument("--file", help="📂 Path to audio file. If not provided, will enter interactive mode.")
    parser.add_argument("--language", help="🌐 Language code (e.g., ms, en). Leave empty to auto-detect")
    parser.add_argument("--output_dir", default=".", help="📁 Directory to save output files")
    parser.add_argument("--formats", default="srt,json", help="💾 Output formats: srt,json,txt (comma-separated)")
    parser.add_argument("--model_size", default="medium", help="Size of the Whisper model (e.g., tiny, base, small, medium, large-v2, large-v3)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to use for computation (auto, cuda, cpu)")
    parser.add_argument("--compute_type", default="auto", help="Compute type for the model (e.g., float16, int8_float16, int8, float32). 'auto' selects float16 for GPU and int8 for CPU.")
    parser.add_argument("--cpu_threads", type=int, default=0, help="Number of CPU threads to use (0 for auto-detection)")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for decoding (e.g., 1 for greedy, 5 for more accuracy)")

    args = parser.parse_args()

    # If no file is provided, enter interactive mode
    if not args.file:
        args = interactive_setup(args)

    # Resolve device and compute type from 'auto'
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    compute_type = args.compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    print(f"{Fore.CYAN}🔊 Loading model '{args.model_size}' on device '{device}' with compute type '{compute_type}'...{Style.RESET_ALL}")

    model_kwargs = {
        "device": device,
        "compute_type": compute_type,
    }
    if device == "cpu":
        model_kwargs["cpu_threads"] = args.cpu_threads

    model = WhisperModel(args.model_size, **model_kwargs)

    language_str = args.language if args.language else "Auto-detect"
    print(f"{Fore.YELLOW}🌍 Language: {language_str}")
    print(f"{Fore.GREEN}📥 File: {args.file}")
    print(f"{Fore.BLUE}📤 Output Dir: {args.output_dir}")
    print(f"{Fore.MAGENTA}💾 Formats: {args.formats.upper()}")
    print(f"{Fore.YELLOW}⚙️  Settings: Device={device}, Compute={compute_type}, Beam Size={args.beam_size}{Style.RESET_ALL}")
    if device == "cpu":
        cpu_threads_str = str(args.cpu_threads) if args.cpu_threads > 0 else "Auto"
        print(f"{Fore.YELLOW}           CPU Threads={cpu_threads_str}{Style.RESET_ALL}\n")
    else:
        print("")

    if args.language is None:
        print(f"{Fore.YELLOW}🤔 Detecting language... (Use --language to force a specific language){Style.RESET_ALL}")

    transcribe_start_time = time.time()
    segments, info = model.transcribe(
        args.file,
        language=args.language,
        beam_size=args.beam_size
    )

    if args.language is None:
        print(f"{Fore.YELLOW}🌍 Detected language: {info.language} (Confidence: {info.language_probability:.2f}){Style.RESET_ALL}")

    segment_list = []
    total_duration = round(info.duration, 2)
    last_pos = 0

    with tqdm(total=total_duration, unit='s', bar_format='{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for segment in segments:
            segment_list.append(segment)
            pbar.update(segment.end - last_pos)
            last_pos = segment.end
        if pbar.n < total_duration: # Ensure bar completes
             pbar.update(total_duration - pbar.n)

    transcribe_end_time = time.time()
    transcribe_time = transcribe_end_time - transcribe_start_time
    transcription_speed = total_duration / transcribe_time if transcribe_time > 0 else 0

    print(f"\n{Fore.YELLOW}🚀 Transcription speed: {transcription_speed:.2f} audio seconds/s{Style.RESET_ALL}")

    base_filename = os.path.splitext(os.path.basename(args.file))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    formats = [fmt.strip().lower() for fmt in args.formats.split(",")]

    print(f"\n{Fore.CYAN}💾 Writing output files...{Style.RESET_ALL}")

    if "srt" in formats:
        srt_path = os.path.join(args.output_dir, base_filename + ".srt")
        write_srt(segment_list, srt_path)
        print(f"{Fore.GREEN}✔ SRT saved: {srt_path}")

    if "json" in formats:
        json_path = os.path.join(args.output_dir, base_filename + ".json")
        write_json(segment_list, json_path)
        print(f"{Fore.GREEN}✔ JSON saved: {json_path}")

    if "txt" in formats:
        txt_path = os.path.join(args.output_dir, base_filename + ".txt")
        write_txt(segment_list, txt_path)
        print(f"{Fore.GREEN}✔ TXT saved: {txt_path}")

    print(f"\n{Fore.CYAN}✅ Done! Subtitles are written to '{os.path.abspath(args.output_dir)}' directory.{Style.RESET_ALL}")

    main_end_time = time.time()
    total_runtime = main_end_time - main_start_time
    print(f"{Fore.CYAN}⏱️  Operation finished in: {timedelta(seconds=total_runtime)}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
