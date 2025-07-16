#!/usr/bin/env python3
"""
Monitor script for Voice Agent WebSocket connections and voice recognition performance.
"""
import asyncio
import websockets
import json
import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoiceAgentMonitor:
    def __init__(self, url="ws://localhost:8000/ws/guest_monitor"):
        self.url = url
        self.stats = {
            'connections': 0,
            'successful_connections': 0,
            'failed_connections': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'audio_chunks_processed': 0,
            'voice_activations': 0,
            'false_positives': 0,
            'connection_drops': 0,
            'average_response_time': 0
        }
        self.response_times = deque(maxlen=100)  # Keep last 100 response times
        self.connection_history = deque(maxlen=50)  # Keep last 50 connection attempts
        self.start_time = time.time()
        
    async def monitor_connection(self, duration_minutes=10):
        """Monitor WebSocket connection for specified duration."""
        logger.info(f"🔍 Starting monitoring for {duration_minutes} minutes...")
        
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time:
            connection_start = time.time()
            self.stats['connections'] += 1
            
            try:
                async with websockets.connect(self.url, timeout=10) as websocket:
                    self.stats['successful_connections'] += 1
                    connection_duration = time.time() - connection_start
                    self.connection_history.append({
                        'timestamp': datetime.now(),
                        'success': True,
                        'duration': connection_duration
                    })
                    
                    logger.info(f"✅ Connected successfully (#{self.stats['successful_connections']})")
                    
                    # Monitor this connection for up to 2 minutes
                    session_end = time.time() + 120
                    
                    while time.time() < session_end and time.time() < end_time:
                        try:
                            # Send periodic test messages
                            test_message = {
                                "type": "ping",
                                "timestamp": datetime.utcnow().isoformat(),
                                "monitor_id": f"monitor_{int(time.time())}"
                            }
                            
                            message_start = time.time()
                            await websocket.send(json.dumps(test_message))
                            self.stats['messages_sent'] += 1
                            
                            # Wait for response
                            try:
                                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                                response_time = (time.time() - message_start) * 1000  # ms
                                self.response_times.append(response_time)
                                
                                data = json.loads(response)
                                self.stats['messages_received'] += 1
                                
                                # Update average response time
                                if self.response_times:
                                    self.stats['average_response_time'] = sum(self.response_times) / len(self.response_times)
                                
                                logger.info(f"📨 Response received in {response_time:.1f}ms - Type: {data.get('type', 'unknown')}")
                                
                                # Track specific message types
                                if data.get('type') == 'audio_processed':
                                    self.stats['audio_chunks_processed'] += 1
                                elif data.get('type') == 'vad_status' and data.get('isActive'):
                                    self.stats['voice_activations'] += 1
                                
                            except asyncio.TimeoutError:
                                logger.warning("⚠️ Response timeout")
                                
                            # Wait before next message
                            await asyncio.sleep(10)
                            
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("❌ Connection closed during session")
                            self.stats['connection_drops'] += 1
                            break
                        except Exception as e:
                            logger.error(f"❌ Error during session: {e}")
                            break
                            
            except Exception as e:
                self.stats['failed_connections'] += 1
                self.connection_history.append({
                    'timestamp': datetime.now(),
                    'success': False,
                    'error': str(e)
                })
                logger.error(f"❌ Connection failed: {e}")
                
            # Wait before next connection attempt
            await asyncio.sleep(30)
    
    def print_stats(self):
        """Print current monitoring statistics."""
        uptime = time.time() - self.start_time
        uptime_str = str(timedelta(seconds=int(uptime)))
        
        print("\n" + "=" * 60)
        print("VOICE AGENT MONITORING REPORT")
        print("=" * 60)
        print(f"Monitoring Duration: {uptime_str}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Connection Statistics
        print("CONNECTION STATISTICS:")
        print(f"  Total Attempts: {self.stats['connections']}")
        print(f"  Successful: {self.stats['successful_connections']}")
        print(f"  Failed: {self.stats['failed_connections']}")
        print(f"  Connection Drops: {self.stats['connection_drops']}")
        
        if self.stats['connections'] > 0:
            success_rate = (self.stats['successful_connections'] / self.stats['connections']) * 100
            print(f"  Success Rate: {success_rate:.1f}%")
        
        print()
        
        # Message Statistics
        print("MESSAGE STATISTICS:")
        print(f"  Messages Sent: {self.stats['messages_sent']}")
        print(f"  Messages Received: {self.stats['messages_received']}")
        print(f"  Average Response Time: {self.stats['average_response_time']:.1f}ms")
        
        if self.response_times:
            min_response = min(self.response_times)
            max_response = max(self.response_times)
            print(f"  Min Response Time: {min_response:.1f}ms")
            print(f"  Max Response Time: {max_response:.1f}ms")
        
        print()
        
        # Voice Recognition Statistics
        print("VOICE RECOGNITION STATISTICS:")
        print(f"  Audio Chunks Processed: {self.stats['audio_chunks_processed']}")
        print(f"  Voice Activations: {self.stats['voice_activations']}")
        print(f"  Estimated False Positives: {self.stats['false_positives']}")
        
        print()
        
        # Recent Connection History
        print("RECENT CONNECTION HISTORY:")
        recent_connections = list(self.connection_history)[-10:]  # Last 10 connections
        for i, conn in enumerate(recent_connections, 1):
            status = "✅ SUCCESS" if conn['success'] else "❌ FAILED"
            timestamp = conn['timestamp'].strftime('%H:%M:%S')
            if conn['success']:
                duration = conn.get('duration', 0)
                print(f"  {i:2d}. [{timestamp}] {status} ({duration:.2f}s)")
            else:
                error = conn.get('error', 'Unknown error')
                print(f"  {i:2d}. [{timestamp}] {status} - {error}")
        
        print("=" * 60)
    
    async def run_monitoring(self, duration_minutes=10):
        """Run the monitoring session."""
        try:
            await self.monitor_connection(duration_minutes)
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring interrupted by user")
        finally:
            self.print_stats()

async def main():
    """Main monitoring function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Voice Agent WebSocket connections')
    parser.add_argument('--duration', type=int, default=10, help='Monitoring duration in minutes (default: 10)')
    parser.add_argument('--url', type=str, default='ws://localhost:8000/ws/guest_monitor', help='WebSocket URL to monitor')
    
    args = parser.parse_args()
    
    monitor = VoiceAgentMonitor(url=args.url)
    await monitor.run_monitoring(duration_minutes=args.duration)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
    except Exception as e:
        logger.error(f"❌ Monitoring failed: {e}")