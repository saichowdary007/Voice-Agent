#!/usr/bin/env python3
import subprocess
import os
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("audio_generator")

# Test files configuration
TEST_FILES = [
    {
        "name": "silence.webm",
        "description": "Silent audio (1 second)",
        "command": [
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "1", 
            "-c:a", "libopus", "-b:a", "32k", "-y"
        ]
    },
    {
        "name": "tone_440hz.webm",
        "description": "440Hz sine wave (1 second)",
        "command": [
            "ffmpeg", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000", "-t", "1", 
            "-c:a", "libopus", "-b:a", "32k", "-y"
        ]
    },
    {
        "name": "tone_1khz.webm",
        "description": "1kHz sine wave (3 seconds)",
        "command": [
            "ffmpeg", "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=16000", "-t", "3", 
            "-c:a", "libopus", "-b:a", "32k", "-y"
        ]
    },
    {
        "name": "white_noise.webm",
        "description": "White noise (2 seconds)",
        "command": [
            "ffmpeg", "-f", "lavfi", "-i", "anoisesrc=color=white:sample_rate=16000:amplitude=0.1", "-t", "2", 
            "-c:a", "libopus", "-b:a", "32k", "-y"
        ]
    },
    {
        "name": "chirp.webm",
        "description": "Chirp signal (sweep from 20Hz to 8kHz)",
        "command": [
            "ffmpeg", "-f", "lavfi", 
            "-i", "sine=frequency=20:sample_rate=16000:beep_factor=4:duration=5", 
            "-af", "asetrate=16000*8000/20", "-t", "5", 
            "-c:a", "libopus", "-b:a", "32k", "-y"
        ]
    },
    {
        "name": "corrupted.webm",
        "description": "Intentionally corrupted WebM file",
        "special": "corrupted"
    }
]

def check_ffmpeg() -> bool:
    """Check if FFmpeg is available"""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"FFmpeg available: {result.stdout.splitlines()[0]}")
            return True
        else:
            logger.error("FFmpeg not available or returned an error")
            return False
    except Exception as e:
        logger.error(f"Error checking FFmpeg: {str(e)}")
        return False

def generate_file(test_file: dict, output_dir: str) -> bool:
    """Generate a test audio file based on configuration"""
    try:
        output_path = os.path.join(output_dir, test_file["name"])
        
        # Special case for corrupted file
        if test_file.get("special") == "corrupted":
            # First create a valid file, then corrupt it
            temp_file = os.path.join(output_dir, "temp.webm")
            cmd = [
                "ffmpeg", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000", 
                "-t", "1", "-c:a", "libopus", "-b:a", "32k", "-y", temp_file
            ]
            
            logger.info(f"Generating base file for corruption: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
                
            # Now corrupt the file by truncating it
            with open(temp_file, "rb") as f:
                data = f.read()
                
            # Write only 80% of the file to create corruption
            with open(output_path, "wb") as f:
                f.write(data[:int(len(data) * 0.8)])
                
            # Remove temp file
            os.remove(temp_file)
            
            logger.info(f"Created corrupted file at {output_path}")
            return True
        
        # Normal file generation
        cmd = test_file["command"] + [output_path]
        
        logger.info(f"Generating {test_file['description']}: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
            
        logger.info(f"Created {test_file['description']} at {output_path}")
        return True
            
    except Exception as e:
        logger.error(f"Error generating test file: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate test audio files for Voice Agent testing")
    parser.add_argument("--output-dir", default="test_audio", help="Output directory for test files")
    parser.add_argument("--file", help="Generate only the specified test file")
    parser.add_argument("--list", action="store_true", help="List available test files")
    args = parser.parse_args()
    
    # List test files if requested
    if args.list:
        print("Available test files:")
        for file in TEST_FILES:
            print(f"  - {file['name']}: {file['description']}")
        return
    
    # Check FFmpeg availability
    if not check_ffmpeg():
        logger.error("FFmpeg is required for this script. Please install it first.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate test files
    if args.file:
        # Generate only the specified file
        for file in TEST_FILES:
            if file["name"] == args.file:
                generate_file(file, args.output_dir)
                break
        else:
            logger.error(f"Test file {args.file} not found. Use --list to see available files.")
    else:
        # Generate all test files
        success_count = 0
        for file in TEST_FILES:
            if generate_file(file, args.output_dir):
                success_count += 1
                
        logger.info(f"Generated {success_count} of {len(TEST_FILES)} test files in {args.output_dir}")

if __name__ == "__main__":
    main() 