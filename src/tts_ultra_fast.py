#!/usr/bin/env python3
"""
Ultra-Fast TTS Module - Production Optimized
Target: <200ms synthesis latency for 500ms total pipeline
"""
import asyncio
import io
import logging
import os
import time
import threading
import subprocess
from typing import Optional, Union, Dict, List
import tempfile
import queue
import base64

# Third-party imports with fallbacks
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logging.warning("⚠️ edge_tts not available - install for optimal performance")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import pydub
    from pydub import AudioSegment
    from pydub.playback import play
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

logger = logging.getLogger(__name__)

class UltraFastTTS:
    """
    Ultra-fast TTS optimized for <200ms synthesis latency
    Features:
    - Pre-warmed audio pipeline
    - Streaming synthesis 
    - Sentence-by-sentence processing
    - Aggressive buffering
    - Local audio processing
    """
    
    def __init__(self):
        self.voice = "en-US-AriaNeural"  # Fastest female voice
        self.rate = "+25%"  # Faster speech rate
        self.volume = "+0%"
        
        # Performance settings
        self.max_chunk_size = 50  # Smaller chunks for faster processing
        self.buffer_size = 2048
        self.streaming_enabled = True
        
        # Pre-warm audio system
        self.audio_initialized = False
        self.temp_dir = tempfile.mkdtemp(prefix="ultra_tts_")
        
        # Performance tracking
        self.synthesis_times = []
        self.last_synthesis_time = 0
        
        # Audio pipeline optimization
        self._initialize_audio_pipeline()
        self._pre_warm_synthesis()
    
    def _initialize_audio_pipeline(self):
        """Pre-initialize audio system for instant playback"""
        try:
            if PYGAME_AVAILABLE:
                pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
                pygame.mixer.init()
                self.audio_initialized = True
                logger.info("✅ Pygame audio initialized")
            elif PYDUB_AVAILABLE:
                # Test pydub audio setup
                silent = AudioSegment.silent(duration=100)
                self.audio_initialized = True
                logger.info("✅ Pydub audio initialized")
            else:
                logger.warning("⚠️ No audio backend available - will use system audio")
        except Exception as e:
            logger.warning(f"⚠️ Audio initialization failed: {e}")
    
    def _pre_warm_synthesis(self):
        """Pre-warm TTS engine for instant response"""
        if not EDGE_TTS_AVAILABLE:
            logger.warning("⚠️ Edge TTS not available - using fallback")
            return
        
        try:
            # Pre-warm with minimal synthesis
            logger.info("🔥 Pre-warming TTS engine...")
            asyncio.run(self._warmup_synthesis())
            logger.info("✅ TTS engine pre-warmed")
        except Exception as e:
            logger.warning(f"⚠️ TTS pre-warm failed: {e}")
    
    async def _warmup_synthesis(self):
        """Perform warmup synthesis"""
        try:
            # Create minimal synthesis to pre-load engine
            communicate = edge_tts.Communicate("Hi", self.voice, rate=self.rate)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
                    break  # Just need first chunk for warmup
        except Exception as e:
            logger.warning(f"⚠️ Warmup synthesis failed: {e}")
    
    def _split_text_optimally(self, text: str) -> List[str]:
        """Split text into optimal chunks for ultra-fast streaming"""
        if len(text) <= self.max_chunk_size:
            return [text]
        
        # Split by sentences first
        import re
        sentences = re.split(r'[.!?]+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed chunk size, start new chunk
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def _synthesize_chunk_streaming(self, text: str) -> bytes:
        """Synthesize single chunk with streaming optimization"""
        if not EDGE_TTS_AVAILABLE:
            return await self._fallback_synthesis(text)
        
        start_time = time.time()
        
        try:
            # Use fastest edge TTS settings
            communicate = edge_tts.Communicate(
                text, 
                self.voice,
                rate=self.rate,
                volume=self.volume
            )
            
            audio_data = b""
            first_chunk_time = None
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        logger.debug(f"🎵 First audio chunk: {(first_chunk_time - start_time) * 1000:.1f}ms")
                    
                    audio_data += chunk["data"]
            
            end_time = time.time()
            synthesis_time = (end_time - start_time) * 1000
            
            # Track performance
            self.synthesis_times.append(synthesis_time)
            self.last_synthesis_time = synthesis_time
            
            logger.debug(f"🎵 Chunk synthesis: {synthesis_time:.1f}ms for {len(text)} chars")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ Chunk synthesis failed: {e}")
            return await self._fallback_synthesis(text)
    
    async def _fallback_synthesis(self, text: str) -> bytes:
        """Fallback synthesis using system TTS"""
        logger.warning(f"⚠️ Using fallback synthesis for: {text[:30]}...")
        
        try:
            # Use macOS say command as fast fallback
            if os.system("which say > /dev/null 2>&1") == 0:
                temp_file = os.path.join(self.temp_dir, f"fallback_{int(time.time() * 1000)}.wav")
                
                # Fast synthesis with say command
                subprocess.run([
                    "say", "-v", "Samantha", "-r", "200", "-o", temp_file, text
                ], check=True, capture_output=True)
                
                # Read audio data
                with open(temp_file, "rb") as f:
                    audio_data = f.read()
                
                # Cleanup
                os.unlink(temp_file)
                
                return audio_data
            
            else:
                # Return empty audio if no TTS available
                logger.warning("⚠️ No TTS backend available")
                return b""
                
        except Exception as e:
            logger.error(f"❌ Fallback synthesis failed: {e}")
            return b""
    
    async def synthesize(self, text: str) -> bytes:
        """
        Ultra-fast text-to-speech synthesis
        Target: <200ms for typical responses
        """
        if not text or not text.strip():
            return b""
        
        start_time = time.time()
        logger.debug(f"🎤 Synthesizing: {text[:50]}...")
        
        try:
            # For very short text, synthesize directly
            if len(text) <= self.max_chunk_size:
                audio_data = await self._synthesize_chunk_streaming(text)
            else:
                # Split into optimal chunks for streaming
                chunks = self._split_text_optimally(text)
                logger.debug(f"📝 Split into {len(chunks)} chunks")
                
                # Synthesize chunks in parallel for speed
                tasks = [self._synthesize_chunk_streaming(chunk) for chunk in chunks]
                chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Combine audio data
                audio_data = b""
                for i, result in enumerate(chunk_results):
                    if isinstance(result, Exception):
                        logger.error(f"❌ Chunk {i} failed: {result}")
                        continue
                    audio_data += result
            
            end_time = time.time()
            total_time = (end_time - start_time) * 1000
            
            # Log performance
            logger.info(f"🎵 TTS synthesis: {total_time:.1f}ms for {len(text)} chars")
            
            # Check if we hit our target
            if total_time <= 200:
                logger.info(f"✅ TTS target achieved: {total_time:.1f}ms ≤ 200ms")
            else:
                logger.warning(f"⚠️ TTS above target: {total_time:.1f}ms > 200ms")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"❌ TTS synthesis failed: {e}")
            return await self._fallback_synthesis(text)
    
    async def synthesize_streaming(self, text: str, callback=None):
        """
        Streaming synthesis with immediate callback for each chunk
        Allows starting playback while synthesis continues
        """
        if not text or not text.strip():
            return
        
        logger.info(f"🎵 Streaming synthesis: {text[:50]}...")
        chunks = self._split_text_optimally(text)
        
        for i, chunk in enumerate(chunks):
            try:
                audio_data = await self._synthesize_chunk_streaming(chunk)
                
                if callback and audio_data:
                    # Call back with chunk audio immediately
                    await callback(audio_data, i, len(chunks))
                
            except Exception as e:
                logger.error(f"❌ Streaming chunk {i} failed: {e}")
    
    def play_audio_optimized(self, audio_data: bytes) -> bool:
        """Play audio with ultra-low latency"""
        if not audio_data:
            return False
        
        try:
            if self.audio_initialized and PYGAME_AVAILABLE:
                # Use pygame for fastest playback
                audio_io = io.BytesIO(audio_data)
                pygame.mixer.music.load(audio_io)
                pygame.mixer.music.play()
                return True
                
            elif PYDUB_AVAILABLE:
                # Use pydub as fallback
                audio_segment = AudioSegment.from_wav(io.BytesIO(audio_data))
                play(audio_segment)
                return True
                
            else:
                # System audio fallback
                temp_file = os.path.join(self.temp_dir, f"audio_{int(time.time() * 1000)}.wav")
                with open(temp_file, "wb") as f:
                    f.write(audio_data)
                
                # Use fastest system audio player
                if os.system("which afplay > /dev/null 2>&1") == 0:  # macOS
                    subprocess.run(["afplay", temp_file], check=True)
                elif os.system("which aplay > /dev/null 2>&1") == 0:  # Linux
                    subprocess.run(["aplay", temp_file], check=True)
                else:
                    logger.warning("⚠️ No audio player available")
                    return False
                
                # Cleanup
                os.unlink(temp_file)
                return True
                
        except Exception as e:
            logger.error(f"❌ Audio playback failed: {e}")
            return False
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        if not self.synthesis_times:
            return {
                "average_latency_ms": 0,
                "last_latency_ms": 0,
                "target_latency_ms": 200,
                "samples": 0
            }
        
        return {
            "average_latency_ms": sum(self.synthesis_times) / len(self.synthesis_times),
            "last_latency_ms": self.last_synthesis_time,
            "min_latency_ms": min(self.synthesis_times),
            "max_latency_ms": max(self.synthesis_times),
            "target_latency_ms": 200,
            "samples": len(self.synthesis_times),
            "target_achieved_ratio": len([t for t in self.synthesis_times if t <= 200]) / len(self.synthesis_times)
        }
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.audio_initialized and PYGAME_AVAILABLE:
                pygame.mixer.quit()
            
            # Cleanup temp directory
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")
    
    def __del__(self):
        """Destructor cleanup"""
        self.cleanup()

# Async context manager for production use
class UltraFastTTSContext:
    """Context manager for ultra-fast TTS with automatic cleanup"""
    
    def __init__(self):
        self.tts = None
    
    async def __aenter__(self):
        self.tts = UltraFastTTS()
        return self.tts
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.tts:
            self.tts.cleanup()

# Factory function for easy instantiation
def create_ultra_fast_tts() -> UltraFastTTS:
    """Create optimized TTS instance"""
    return UltraFastTTS()

# Test function for performance validation
async def test_performance():
    """Test TTS performance against 200ms target"""
    tts = UltraFastTTS()
    
    test_texts = [
        "Hello world",
        "How can I help you today?",
        "The weather is quite nice today, isn't it?",
        "I understand your question and I'm happy to help you with that specific request."
    ]
    
    results = []
    
    for text in test_texts:
        start_time = time.time()
        audio_data = await tts.synthesize(text)
        end_time = time.time()
        
        latency = (end_time - start_time) * 1000
        results.append({
            "text": text,
            "latency_ms": latency,
            "target_achieved": latency <= 200,
            "audio_size": len(audio_data)
        })
    
    # Print results
    print("\n🎵 Ultra-Fast TTS Performance Test")
    print("=" * 50)
    for result in results:
        status = "✅" if result["target_achieved"] else "❌"
        print(f"{status} {result['latency_ms']:.1f}ms - {result['text'][:30]}...")
    
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    target_achieved = sum(1 for r in results if r["target_achieved"]) / len(results)
    
    print(f"\n📊 Average latency: {avg_latency:.1f}ms")
    print(f"🎯 Target achieved: {target_achieved * 100:.1f}%")
    
    tts.cleanup()
    return results

if __name__ == "__main__":
    # Run performance test
    asyncio.run(test_performance()) 