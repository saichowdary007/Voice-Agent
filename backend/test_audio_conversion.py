#!/usr/bin/env python3
import os
import sys
import asyncio
import tempfile
import subprocess

async def convert_to_pcm(audio_data, debug=True):
    """Convert WebM/Ogg audio to raw PCM data"""
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as in_file, \
             tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as out_file:
            
            in_path = in_file.name
            out_path = out_file.name
            
            # Write input data
            in_file.write(audio_data)
            in_file.flush()
            
            # Convert using ffmpeg
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-i', in_path,  # Input file
                '-ar', '16000',  # Sample rate
                '-ac', '1',  # Mono
                '-f', 's16le',  # 16-bit PCM
                out_path  # Output file
            ]
            
            if debug:
                print(f"Running FFmpeg command: {' '.join(cmd)}")
            
            # Run ffmpeg
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await process.communicate()
            
            if process.returncode != 0:
                error = stderr.decode() if stderr else f"FFmpeg error code: {process.returncode}"
                print(f"FFmpeg conversion error: {error}")
                return None
            
            # Read output PCM data
            with open(out_path, 'rb') as f:
                pcm_data = f.read()
            
            # Clean up temp files
            try:
                os.unlink(in_path)
                os.unlink(out_path)
            except Exception as e:
                print(f"Failed to clean up temp files: {e}")
            
            return pcm_data
            
    except Exception as e:
        print(f"Audio conversion error: {e}")
        return None

def check_audio_format(audio_data):
    """Check audio format"""
    is_webm = audio_data.startswith(b'\x1a\x45\xdf\xa3')  # WebM magic bytes
    is_wav = audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:20]  # WAV magic bytes
    is_ogg = audio_data.startswith(b'OggS')  # Ogg magic bytes
    
    if is_webm:
        return "WebM"
    elif is_wav:
        return "WAV"
    elif is_ogg:
        return "OGG"
    else:
        # Try to detect PCM by checking byte patterns
        pcm_bytes = audio_data[:20]
        if all(0 <= b <= 255 for b in pcm_bytes) and any(b > 0 for b in pcm_bytes):
            # Get first few bytes for debug
            first_bytes = ", ".join([f"{b:02x}" for b in pcm_bytes[:10]])
            return f"Likely PCM (first bytes: {first_bytes})"
        
        return "Unknown"

async def main():
    test_file = "../test_audio/tone_440hz.webm"
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        
        # Check if dummy.webm exists
        dummy_file = "../dummy.webm"
        if os.path.exists(dummy_file):
            test_file = dummy_file
            print(f"Using dummy file: {dummy_file}")
        else:
            print("No test files found. Exiting.")
            return
    
    print(f"Testing audio conversion with: {test_file}")
    
    with open(test_file, 'rb') as f:
        audio_data = f.read()
    
    format_name = check_audio_format(audio_data)
    print(f"Detected format: {format_name}")
    print(f"File size: {len(audio_data)} bytes")
    
    print("Converting to PCM...")
    pcm_data = await convert_to_pcm(audio_data)
    
    if pcm_data:
        print(f"Conversion successful. PCM data size: {len(pcm_data)} bytes")
        
        # Save PCM data for debugging
        with open('test_output.pcm', 'wb') as f:
            f.write(pcm_data)
        print("Saved PCM data to test_output.pcm")
    else:
        print("Conversion failed.")

if __name__ == "__main__":
    asyncio.run(main()) 