#!/usr/bin/env python3
"""
==============================================================================
 Media Processor Module — Google Colab & Local
 Version: 1.0.0
 
 Pre-processes video/audio files BEFORE loading heavy AI models:
   - Smart Bitrate Analyzer (prevents size inflation on hyper-compressed video)
   - Fast FFmpeg H.265 (libx265) video compression
   - Super-fast MP3 audio extraction (64 kbps mono)
   - Google Drive integration
==============================================================================
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path

SUPPORTED_EXTS = ('.mp4', '.mkv', '.avi', '.mov', '.webm', '.mp3',
                 '.wav', '.m4a', '.flac', '.ogg', '.aac')


def get_video_info(filepath):
    """
    Uses ffprobe to extract duration (secs), resolution (height), and bitrate (kbps).
    Returns dict or None if probe fails.
    """
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_format', '-show_streams',
        filepath
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        format_info = data.get('format', {})
        duration = float(format_info.get('duration', 0))
        size_bytes = int(format_info.get('size', 0))
        
        # Calculate overall bitrate in kbps
        bitrate_kbps = (size_bytes * 8 / 1024) / duration if duration > 0 else 0
        
        # Find video stream height
        height = 720  # default assumption
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                height = int(stream.get('height', 720))
                break

        return {
            'duration': duration,
            'size_mb': size_bytes / (1024 * 1024),
            'bitrate_kbps': bitrate_kbps,
            'height': height
        }
    except Exception as e:
        print(f"   ⚠️ ffprobe warning for {os.path.basename(filepath)}: {e}")
        return None


def should_compress_video(info):
    """
    Smart Analyzer Decision Rule:
    Determines if video compression will actually REDUCE file size or INCREASE it.
    """
    if not info:
        return True, 30  # Default to compress at CRF 30
        
    height = info['height']
    bitrate = info['bitrate_kbps']
    
    if height <= 480:
        if bitrate < 350:
            return False, 0  # Skip: already hyper-compressed!
        return True, 32
    elif height <= 720:
        if bitrate < 450:
            return False, 0  # Skip: already hyper-compressed!
        return True, 31
    else:  # 1080p+
        if bitrate < 650:
            return False, 0  # Skip: already low bitrate!
        return True, 29


def extract_audio(input_path, output_path):
    """
    Extracts audio to 64 kbps mono MP3 using FFmpeg (super fast, tiny file size).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vn', '-c:a', 'libmp3lame',
        '-b:a', '64k', '-ac', '1',
        output_path
    ]
    print(f"   ⚡ Extracting Audio (MP3 64k Mono): {os.path.basename(input_path)}")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    print(f"   ✅ Saved MP3: {output_path} ({os.path.getsize(output_path)/(1024*1024):.1f} MB)")
    return output_path


def is_nvenc_available():
    """
    Checks if NVIDIA NVENC hardware video encoder is supported by FFmpeg and GPU.
    """
    try:
        cmd = ['ffmpeg', '-f', 'lavfi', '-i', 'nullsrc', '-c:v', 'hevc_nvenc', '-frames:v', '1', '-f', 'null', '-']
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return res.returncode == 0
    except Exception:
        return False


def compress_video(input_path, output_path, crf=30):
    """
    Compresses video using FFmpeg.
    Uses 10x faster NVIDIA GPU NVENC (hevc_nvenc) if GPU available,
    otherwise falls back to standard CPU software (libx265).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    use_gpu = is_nvenc_available()
    
    if use_gpu:
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'hevc_nvenc', '-rc:v', 'vbr', '-cq:v', str(crf), '-b:v', '0', '-preset', 'p4',
            '-c:a', 'aac', '-b:a', '96k', '-ac', '1',
            output_path
        ]
        print(f"   ⚡ Compressing Video [NVIDIA GPU NVENC - 10x Fast]: {os.path.basename(input_path)}")
    else:
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx265', '-crf', str(crf), '-preset', 'fast',
            '-c:a', 'aac', '-b:a', '96k', '-ac', '1',
            output_path
        ]
        print(f"   🎬 Compressing Video [CPU x265 CRF {crf}]: {os.path.basename(input_path)}")
        
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    orig_mb = os.path.getsize(input_path) / (1024 * 1024)
    new_mb = os.path.getsize(output_path) / (1024 * 1024)

    # Ultimate Safety Check: Never keep a file that became larger!
    if new_mb >= orig_mb:
        print(f"   ⚠️ Re-encoded file was larger ({new_mb:.1f} MB >= {orig_mb:.1f} MB). Keeping original!")
        try:
            os.remove(output_path)
        except Exception:
            pass
        return input_path

    print(f"   ✅ Compressed: {orig_mb:.1f} MB ➔ {new_mb:.1f} MB ({output_path})")
    return output_path


def process_media_file(filepath, compressed_dir, mode='audio'):
    """
    Main processing entry for a single file.
      mode='audio': Extracts audio to MP3 (Recommended)
      mode='video': Smart Video Compression
      mode='none': Copy as-is to compressed folder
    """
    filename = os.path.basename(filepath)
    name_no_ext, ext = os.path.splitext(filename)
    os.makedirs(compressed_dir, exist_ok=True)
    
    if mode == 'audio':
        output_path = os.path.join(compressed_dir, f"{name_no_ext}.mp3")
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"   ⏩ Audio already exists: {output_path}")
            return output_path
        return extract_audio(filepath, output_path)

    elif mode == 'video':
        output_path = os.path.join(compressed_dir, f"{name_no_ext}.mp4")
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"   ⏩ Compressed video already exists: {output_path}")
            return output_path
            
        info = get_video_info(filepath)
        should_compress, crf = should_compress_video(info)
        
        if not should_compress:
            print(f"   ℹ️  Skipping compression (video already low bitrate ~{info['bitrate_kbps']:.0f} kbps): {filename}")
            return filepath  # Return original filepath directly, zero copy needed!
        else:
            return compress_video(filepath, output_path, crf=crf)

    else:
        # No compression mode: use original directly
        return filepath
