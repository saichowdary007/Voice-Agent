#!/usr/bin/env python3
"""
Download required models for the voice agent backend
"""
import os
import sys
import urllib.request
import tarfile
import zipfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model URLs and configurations
MODELS = {
    "sherpa_ncnn": {
        "url": "https://github.com/k2-fsa/sherpa-ncnn/releases/download/models/sherpa-ncnn-streaming-zipformer-20M-2023-02-17.tar.bz2",
        "extract_dir": "sherpa-ncnn-streaming-zipformer-20M-2023-02-17",
        "files": [
            "encoder_jit_trace-pnnx.ncnn.param",
            "encoder_jit_trace-pnnx.ncnn.bin",
            "decoder_jit_trace-pnnx.ncnn.param", 
            "decoder_jit_trace-pnnx.ncnn.bin",
            "joiner_jit_trace-pnnx.ncnn.param",
            "joiner_jit_trace-pnnx.ncnn.bin",
            "tokens.txt"
        ]
    },
    "piper_tts": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts/high/en_US-libritts-high.onnx",
        "config_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts/high/en_US-libritts-high.onnx.json",
        "files": [
            "en_US-libritts-high.onnx",
            "en_US-libritts-high.onnx.json"
        ]
    }
}

def download_file(url: str, destination: Path, show_progress: bool = True):
    """Download a file with progress indication"""
    try:
        def progress_hook(block_num, block_size, total_size):
            if show_progress and total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded * 100) // total_size)
                sys.stdout.write(f"\rDownloading: {percent}% ({downloaded}/{total_size} bytes)")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(url, destination, progress_hook)
        if show_progress:
            print()  # New line after progress
        logger.info(f"Downloaded: {destination.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False

def extract_archive(archive_path: Path, extract_to: Path):
    """Extract tar.bz2 or zip archive"""
    try:
        if archive_path.suffix == '.bz2':
            with tarfile.open(archive_path, 'r:bz2') as tar:
                tar.extractall(extract_to)
        elif archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_file:
                zip_file.extractall(extract_to)
        else:
            logger.error(f"Unsupported archive format: {archive_path}")
            return False
            
        logger.info(f"Extracted: {archive_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to extract {archive_path}: {e}")
        return False

def download_sherpa_ncnn_models(model_dir: Path):
    """Download and extract sherpa-ncnn models"""
    logger.info("Downloading sherpa-ncnn Zipformer-20M models...")
    
    config = MODELS["sherpa_ncnn"]
    archive_name = "sherpa-ncnn-zipformer.tar.bz2"
    archive_path = model_dir / archive_name
    
    # Download archive
    if not download_file(config["url"], archive_path):
        return False
    
    # Extract archive
    if not extract_archive(archive_path, model_dir):
        return False
    
    # Move files to model_dir root
    extract_dir = model_dir / config["extract_dir"]
    if extract_dir.exists():
        for file_name in config["files"]:
            src_file = extract_dir / file_name
            dst_file = model_dir / file_name
            
            if src_file.exists():
                src_file.rename(dst_file)
                logger.info(f"Moved: {file_name}")
            else:
                logger.warning(f"Expected file not found: {file_name}")
        
        # Clean up
        try:
            import shutil
            shutil.rmtree(extract_dir)
            archive_path.unlink()
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    return True

def download_piper_models(model_dir: Path):
    """Download Piper TTS models"""
    logger.info("Downloading Piper TTS en_US-libritts-high models...")
    
    config = MODELS["piper_tts"]
    
    # Download model file
    model_file = model_dir / "en_US-libritts-high.onnx"
    if not download_file(config["url"], model_file):
        return False
    
    # Download config file
    config_file = model_dir / "en_US-libritts-high.onnx.json"
    if not download_file(config["config_url"], config_file):
        return False
    
    return True

def verify_models(model_dir: Path):
    """Verify that all required model files exist"""
    logger.info("Verifying downloaded models...")
    
    required_files = []
    
    # Add sherpa-ncnn files
    required_files.extend(MODELS["sherpa_ncnn"]["files"])
    
    # Add Piper files
    required_files.extend(MODELS["piper_tts"]["files"])
    
    missing_files = []
    for file_name in required_files:
        file_path = model_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
        else:
            file_size = file_path.stat().st_size
            logger.info(f"✅ {file_name} ({file_size} bytes)")
    
    if missing_files:
        logger.error(f"Missing model files: {missing_files}")
        return False
    
    logger.info("✅ All model files verified successfully")
    return True

def main():
    """Main download function"""
    # Get model directory from environment or use default
    model_dir = Path(os.getenv("MODEL_PATH", "/app/models"))
    
    # Create model directory
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Model directory: {model_dir}")
    
    # Check if models already exist
    if verify_models(model_dir):
        logger.info("Models already exist and verified. Skipping download.")
        return 0
    
    try:
        # Download sherpa-ncnn models
        if not download_sherpa_ncnn_models(model_dir):
            logger.error("Failed to download sherpa-ncnn models")
            return 1
        
        # Download Piper models
        if not download_piper_models(model_dir):
            logger.error("Failed to download Piper models")
            return 1
        
        # Final verification
        if not verify_models(model_dir):
            logger.error("Model verification failed")
            return 1
        
        logger.info("🎉 All models downloaded and verified successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Download process failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 