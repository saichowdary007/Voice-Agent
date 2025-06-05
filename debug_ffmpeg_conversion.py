#!/usr/bin/env python3
import subprocess
import tempfile
import os
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ffmpeg_debug")

# Configuration
DEFAULT_WEBM_OUTPUT = "valid_test_audio.webm"

def generate_valid_webm(output_path: str, duration: int = 5) -> bool:
    """Generate a valid WebM/Opus file using FFmpeg"""
    try:
        # Create a sine wave audio file using FFmpeg
        cmd = [
            "ffmpeg", 
            "-f", "lavfi", 
            "-i", f"sine=frequency=440:duration={duration}", 
            "-c:a", "libopus", 
            "-b:a", "32k",
            "-y",  # Overwrite output file
            output_path
        ]
        
        logger.info(f"Generating valid WebM file: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False
        
        logger.info(f"Successfully created valid WebM file at {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating WebM file: {str(e)}")
        return False

def extract_webm_header(webm_file: str, output_file: str = "webm_header.bin") -> bool:
    """Extract the header from a valid WebM file"""
    try:
        # Extract first 256 bytes which should contain the header
        with open(webm_file, "rb") as f:
            header_data = f.read(256)
        
        # Save header to file
        with open(output_file, "wb") as f:
            f.write(header_data)
            
        # Also display header as hex for copying into code
        hex_data = header_data.hex()
        formatted_hex = ' '.join(hex_data[i:i+2] for i in range(0, len(hex_data), 2))
        
        logger.info(f"WebM header hex (first 256 bytes):")
        
        # Format for code insertion, 16 bytes per line
        hex_lines = []
        for i in range(0, len(hex_data), 32):
            line = hex_data[i:i+32]
            formatted_line = '"' + ''.join('\\x' + line[j:j+2] for j in range(0, len(line), 2)) + '"'
            hex_lines.append(formatted_line)
        
        for line in hex_lines:
            print(line + " +")
        
        logger.info(f"Saved header to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error extracting WebM header: {str(e)}")
        return False

def analyze_webm_file(webm_file: str) -> bool:
    """Analyze a WebM file using FFprobe"""
    try:
        cmd = [
            "ffprobe", 
            "-i", webm_file,
            "-show_format",
            "-show_streams",
            "-v", "error"
        ]
        
        logger.info(f"Analyzing WebM file: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFprobe error: {result.stderr}")
            return False
        
        logger.info("FFprobe analysis:")
        print(result.stdout)
        return True
        
    except Exception as e:
        logger.error(f"Error analyzing WebM file: {str(e)}")
        return False

def debug_conversion(input_file: str) -> bool:
    """Debug conversion of a problematic file"""
    try:
        # Try different conversion methods
        methods = [
            ["-f", "matroska"],
            ["-f", "webm"],
            ["-f", "opus"],
            []  # No format specification
        ]
        
        for method in methods:
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
                cmd = ["ffmpeg", "-i", input_file] + method + ["-y", temp_file.name]
                
                logger.info(f"Trying conversion with: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info(f"✅ Conversion successful with {method}")
                    logger.info(f"Output file size: {os.path.getsize(temp_file.name)} bytes")
                else:
                    logger.error(f"❌ Conversion failed with {method}")
                    logger.error(f"Error: {result.stderr}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during conversion debugging: {str(e)}")
        return False

def check_ffmpeg_availability() -> bool:
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

def main():
    parser = argparse.ArgumentParser(description="Debug FFmpeg conversion issues")
    parser.add_argument("--generate", action="store_true", help="Generate a valid WebM test file")
    parser.add_argument("--output", default=DEFAULT_WEBM_OUTPUT, help="Output file path for generated WebM")
    parser.add_argument("--extract-header", action="store_true", help="Extract and analyze WebM header")
    parser.add_argument("--analyze", action="store_true", help="Analyze WebM file structure")
    parser.add_argument("--debug-conversion", help="Debug conversion of a problematic file")
    args = parser.parse_args()
    
    # Check FFmpeg availability
    if not check_ffmpeg_availability():
        logger.error("FFmpeg is required for this script. Please install it first.")
        return
    
    # Run requested operations
    if args.generate:
        generate_valid_webm(args.output)
    
    if args.extract_header:
        if not os.path.exists(args.output):
            logger.error(f"File not found: {args.output}. Generate it first with --generate")
            return
        extract_webm_header(args.output)
    
    if args.analyze:
        if not os.path.exists(args.output):
            logger.error(f"File not found: {args.output}. Generate it first with --generate")
            return
        analyze_webm_file(args.output)
    
    if args.debug_conversion:
        if not os.path.exists(args.debug_conversion):
            logger.error(f"File not found: {args.debug_conversion}")
            return
        debug_conversion(args.debug_conversion)

if __name__ == "__main__":
    main() 