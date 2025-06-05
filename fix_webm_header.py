#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import logging
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("webm_header_fix")

def find_backend_files(backend_dir, pattern):
    """Find files in backend directory matching pattern"""
    matches = []
    for root, dirs, files in os.walk(backend_dir):
        for file in files:
            if file.endswith('.py') and pattern.search(file):
                matches.append(os.path.join(root, file))
    return matches

def extract_webm_header_from_code(file_path):
    """Extract WebM header definitions from Python code"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Look for WebM header definitions
        # Pattern matches Python byte literals or hex strings that might be headers
        patterns = [
            r'b[\'"](.*?)[\'"]',  # byte literals
            r'bytes\.fromhex\([\'"]([0-9a-fA-F]+)[\'"]\)',  # hex strings
            r'bytes\(\[(.*?)\]\)',  # byte arrays
        ]
        
        headers = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches:
                headers.extend(matches)
                
        return headers
    except Exception as e:
        logger.error(f"Error extracting headers from {file_path}: {str(e)}")
        return []

def check_valid_webm_header(header_bytes):
    """Check if the header bytes represent a valid WebM header"""
    # A valid WebM header should start with the EBML signature
    # 0x1A 0x45 0xDF 0xA3
    if isinstance(header_bytes, str):
        # Convert hex string to bytes
        try:
            # Check if it's a byte literal
            if header_bytes.startswith('\\x'):
                # Convert Python byte literal
                header_bytes = bytes.fromhex(header_bytes.replace('\\x', ''))
            else:
                # Assume it's a hex string
                header_bytes = bytes.fromhex(header_bytes)
        except Exception as e:
            logger.error(f"Error converting header to bytes: {str(e)}")
            return False
    
    # Check for EBML signature
    return (len(header_bytes) >= 4 and 
            header_bytes[0] == 0x1A and 
            header_bytes[1] == 0x45 and 
            header_bytes[2] == 0xDF and 
            header_bytes[3] == 0xA3)

def get_valid_webm_header():
    """Generate a valid WebM header using ffmpeg and return as bytes"""
    try:
        # First try to use an existing valid file
        if os.path.exists('valid_test_audio.webm'):
            with open('valid_test_audio.webm', 'rb') as f:
                header = f.read(256)  # First 256 bytes should include header
            return header
        
        # If no file exists, generate one
        temp_file = 'temp_header.webm'
        cmd = [
            "ffmpeg", 
            "-f", "lavfi", 
            "-i", "sine=frequency=440:duration=1", 
            "-c:a", "libopus", 
            "-b:a", "32k",
            "-y",
            temp_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return None
            
        with open(temp_file, 'rb') as f:
            header = f.read(256)  # First 256 bytes should include header
            
        # Clean up
        os.remove(temp_file)
        
        return header
    except Exception as e:
        logger.error(f"Error generating valid WebM header: {str(e)}")
        return None

def format_header_for_code(header_bytes, format_type='bytes_array'):
    """Format header bytes for insertion into code"""
    if format_type == 'bytes_array':
        # Format as bytes array: bytes([0x1A, 0x45, ...])
        hex_values = [f"0x{b:02X}" for b in header_bytes]
        # Format with 8 bytes per line
        lines = []
        for i in range(0, len(hex_values), 8):
            line = ", ".join(hex_values[i:i+8])
            lines.append(line)
        
        return "bytes([\n    " + ",\n    ".join(lines) + "\n])"
        
    elif format_type == 'hex_string':
        # Format as hex string: bytes.fromhex("1a45dfa3...")
        hex_string = ''.join([f"{b:02x}" for b in header_bytes])
        # Format with 32 chars per line
        lines = []
        for i in range(0, len(hex_string), 32):
            lines.append(hex_string[i:i+32])
        
        return 'bytes.fromhex(\n    "' + '"\n    "'.join(lines) + '"\n)'
        
    elif format_type == 'byte_literal':
        # Format as byte literal: b"\x1A\x45\xDF\xA3..."
        byte_literal = ''.join([f"\\x{b:02x}" for b in header_bytes])
        # Format with 16 bytes per line
        lines = []
        for i in range(0, len(byte_literal), 32):
            lines.append(byte_literal[i:i+32])
        
        return 'b"' + '"\n    b"'.join(lines) + '"'
    
    else:
        raise ValueError(f"Unknown format type: {format_type}")

def suggest_fixes(file_path, invalid_headers, valid_header):
    """Suggest fixes for invalid WebM headers in the file"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # Find line numbers for each invalid header
        header_lines = []
        for header in invalid_headers:
            for i, line in enumerate(lines):
                if header in line:
                    header_lines.append((i, header))
                    
        # Print suggestions
        if header_lines:
            print(f"\nSuggested fixes for {file_path}:")
            print("=" * 50)
            
            for line_num, header in header_lines:
                print(f"Invalid header found at line {line_num + 1}:")
                print(f"  {lines[line_num].strip()}")
                print("\nReplace with:")
                
                # Determine format based on the original
                if 'bytes.fromhex' in lines[line_num]:
                    format_type = 'hex_string'
                elif 'bytes([' in lines[line_num] or 'bytes(' in lines[line_num]:
                    format_type = 'bytes_array'
                else:
                    format_type = 'byte_literal'
                    
                replacement = format_header_for_code(valid_header, format_type)
                print(f"  {replacement}")
                print("\n" + "-" * 50)
        else:
            print(f"No specific line numbers found for invalid headers in {file_path}")
            
    except Exception as e:
        logger.error(f"Error suggesting fixes: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Fix WebM headers in Voice Agent backend code")
    parser.add_argument("--backend-dir", default=".", help="Backend code directory")
    parser.add_argument("--scan", action="store_true", help="Scan backend code for WebM headers")
    parser.add_argument("--check", help="Check specific file for WebM headers")
    parser.add_argument("--generate", action="store_true", help="Generate valid WebM header formats")
    args = parser.parse_args()
    
    # Generate valid WebM header
    valid_header = get_valid_webm_header()
    if not valid_header:
        logger.error("Failed to generate valid WebM header. Make sure FFmpeg is installed.")
        return
        
    if args.generate:
        # Print valid header in different formats
        print("\nValid WebM Header Formats:")
        print("=" * 50)
        print("\nAs bytes array:")
        print(format_header_for_code(valid_header, 'bytes_array'))
        
        print("\nAs hex string:")
        print(format_header_for_code(valid_header, 'hex_string'))
        
        print("\nAs byte literal:")
        print(format_header_for_code(valid_header, 'byte_literal'))
        
        return
    
    # Check specific file
    if args.check:
        if not os.path.exists(args.check):
            logger.error(f"File not found: {args.check}")
            return
            
        logger.info(f"Checking {args.check} for WebM headers...")
        headers = extract_webm_header_from_code(args.check)
        
        if not headers:
            logger.info("No WebM headers found in the file.")
            return
            
        # Check each header
        invalid_headers = []
        for header in headers:
            if not check_valid_webm_header(header):
                invalid_headers.append(header)
                
        if invalid_headers:
            logger.warning(f"Found {len(invalid_headers)} invalid WebM headers in {args.check}")
            suggest_fixes(args.check, invalid_headers, valid_header)
        else:
            logger.info(f"All WebM headers in {args.check} seem valid.")
        
        return
    
    # Scan backend code
    if args.scan:
        logger.info(f"Scanning {args.backend_dir} for files with WebM headers...")
        
        # Common file patterns for audio processing
        patterns = [
            re.compile(r'audio'),
            re.compile(r'webm'),
            re.compile(r'opus'),
            re.compile(r'convert')
        ]
        
        all_files = []
        for pattern in patterns:
            files = find_backend_files(args.backend_dir, pattern)
            all_files.extend(files)
            
        # Remove duplicates
        all_files = list(set(all_files))
        
        if not all_files:
            logger.info("No relevant files found.")
            return
            
        logger.info(f"Found {len(all_files)} potentially relevant files.")
        
        # Check each file for WebM headers
        files_with_headers = []
        for file in all_files:
            headers = extract_webm_header_from_code(file)
            if headers:
                invalid_headers = []
                for header in headers:
                    if not check_valid_webm_header(header):
                        invalid_headers.append(header)
                        
                if invalid_headers:
                    files_with_headers.append((file, invalid_headers))
                    
        if files_with_headers:
            logger.warning(f"Found {len(files_with_headers)} files with invalid WebM headers:")
            for file, headers in files_with_headers:
                logger.warning(f"  - {file} ({len(headers)} invalid headers)")
                suggest_fixes(file, headers, valid_header)
        else:
            logger.info("No files with invalid WebM headers found.")
    
    # Default behavior if no specific action
    if not args.scan and not args.check and not args.generate:
        logger.error("Please specify an action: --scan, --check, or --generate")
        parser.print_help()

if __name__ == "__main__":
    main() 