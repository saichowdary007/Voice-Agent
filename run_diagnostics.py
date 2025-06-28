#!/usr/bin/env python3
"""
Main diagnostic runner script.
Orchestrates all WebSocket and voice pipeline diagnostics.
"""

import asyncio
import json
import logging
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Import our diagnostic modules
from websocket_diagnostics import WebSocketDiagnostics
from websocket_monitor import RealTimeMonitor
from test_voice_pipeline import VoicePipelineTester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiagnosticRunner:
    """Orchestrates all diagnostic tests and monitoring."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.results_dir = Path("diagnostic_results")
        self.results_dir.mkdir(exist_ok=True)
        
    async def run_quick_check(self) -> dict:
        """Run a quick health check of the system."""
        logger.info("🚀 Running quick system health check...")
        
        diagnostics = WebSocketDiagnostics(self.base_url)
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "test_type": "quick_check",
            "base_url": self.base_url
        }
        
        # Test basic connectivity
        connectivity = await diagnostics.test_websocket_connectivity()
        results["websocket_connectivity"] = connectivity
        
        # Test authentication
        auth = await diagnostics.test_authenticated_connection()
        results["authentication"] = auth
        
        # Quick latency test
        latency = await diagnostics.measure_latency_pipeline(num_tests=1)
        results["latency"] = latency
        
        # Determine overall health
        health_score = 0
        if connectivity.get("handshake_successful"):
            health_score += 1
        if auth.get("websocket_auth_successful"):
            health_score += 1
        if latency.get("successful_tests", 0) > 0:
            health_score += 1
        
        results["health_score"] = health_score
        results["health_status"] = "GOOD" if health_score >= 2 else "POOR"
        
        return results
    
    async def run_comprehensive_diagnostics(self) -> dict:
        """Run comprehensive diagnostics including voice pipeline tests."""
        logger.info("🔍 Running comprehensive diagnostics...")
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "test_type": "comprehensive",
            "base_url": self.base_url
        }
        
        # WebSocket diagnostics
        logger.info("Running WebSocket diagnostics...")
        diagnostics = WebSocketDiagnostics(self.base_url)
        ws_results = await diagnostics.run_comprehensive_diagnostics()
        results["websocket_diagnostics"] = ws_results
        
        # Voice pipeline tests
        logger.info("Running voice pipeline tests...")
        voice_tester = VoicePipelineTester(self.base_url)
        voice_results = await voice_tester.run_comprehensive_tests()
        results["voice_pipeline_tests"] = voice_results
        
        # Calculate overall health
        ws_success_rate = ws_results.get("test_summary", {}).get("passed_tests", 0) / max(1, ws_results.get("test_summary", {}).get("total_tests", 1))
        voice_success_rate = voice_results.get("test_summary", {}).get("overall_success_rate", 0)
        
        overall_success_rate = (ws_success_rate + voice_success_rate) / 2
        results["overall_success_rate"] = overall_success_rate
        
        if overall_success_rate >= 0.8:
            results["overall_health"] = "EXCELLENT"
        elif overall_success_rate >= 0.6:
            results["overall_health"] = "GOOD"
        elif overall_success_rate >= 0.4:
            results["overall_health"] = "FAIR"
        else:
            results["overall_health"] = "POOR"
        
        return results
    
    async def run_monitoring_session(self, duration: int = 300) -> dict:
        """Run real-time monitoring session."""
        logger.info(f"📊 Starting monitoring session for {duration} seconds...")
        
        monitor = RealTimeMonitor(self.base_url)
        
        try:
            await monitor.start_monitoring(duration)
            results = monitor.generate_report()
            results["test_type"] = "monitoring"
            results["monitoring_duration_seconds"] = duration
            return results
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring interrupted by user")
            results = monitor.generate_report()
            results["test_type"] = "monitoring_interrupted"
            return results
    
    def save_results(self, results: dict, test_type: str) -> str:
        """Save results to a timestamped file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"{test_type}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📄 Results saved to: {filename}")
        return str(filename)
    
    def print_summary(self, results: dict):
        """Print a summary of the diagnostic results."""
        print("\n" + "="*60)
        print("🔍 DIAGNOSTIC SUMMARY")
        print("="*60)
        
        test_type = results.get("test_type", "unknown")
        print(f"Test Type: {test_type}")
        print(f"Timestamp: {results.get('timestamp', 'N/A')}")
        print(f"Base URL: {results.get('base_url', 'N/A')}")
        print()
        
        if test_type == "quick_check":
            health_status = results.get("health_status", "UNKNOWN")
            health_score = results.get("health_score", 0)
            print(f"Health Status: {health_status}")
            print(f"Health Score: {health_score}/3")
            
            # Connectivity
            connectivity = results.get("websocket_connectivity", {})
            if connectivity.get("handshake_successful"):
                print("✅ WebSocket Connectivity: PASS")
            else:
                print("❌ WebSocket Connectivity: FAIL")
                print(f"   Error: {connectivity.get('error_message', 'Unknown')}")
            
            # Authentication
            auth = results.get("authentication", {})
            if auth.get("websocket_auth_successful"):
                print("✅ Authentication: PASS")
            else:
                print("❌ Authentication: FAIL")
                print(f"   Error: {auth.get('error_message', 'Unknown')}")
            
            # Latency
            latency = results.get("latency", {})
            if latency.get("successful_tests", 0) > 0:
                avg_latency = latency.get("average_latency_ms", 0)
                print(f"✅ Latency Test: PASS ({avg_latency:.1f}ms average)")
            else:
                print("❌ Latency Test: FAIL")
        
        elif test_type == "comprehensive":
            overall_health = results.get("overall_health", "UNKNOWN")
            success_rate = results.get("overall_success_rate", 0)
            print(f"Overall Health: {overall_health}")
            print(f"Success Rate: {success_rate:.1%}")
            
            # WebSocket diagnostics
            ws_results = results.get("websocket_diagnostics", {})
            ws_summary = ws_results.get("test_summary", {})
            ws_passed = ws_summary.get("passed_tests", 0)
            ws_total = ws_summary.get("total_tests", 0)
            print(f"WebSocket Tests: {ws_passed}/{ws_total} passed")
            
            # Voice pipeline tests
            voice_results = results.get("voice_pipeline_tests", {})
            voice_success = voice_results.get("test_summary", {}).get("overall_success_rate", 0)
            print(f"Voice Pipeline: {voice_success:.1%} success rate")
        
        elif test_type in ["monitoring", "monitoring_interrupted"]:
            duration = results.get("monitoring_duration_seconds", 0)
            print(f"Monitoring Duration: {duration} seconds")
            
            # Connection stats
            conn_stats = results.get("connection_stats", {})
            total_connections = conn_stats.get("total_connections", 0)
            avg_connect_time = conn_stats.get("average_connect_time_ms", 0)
            print(f"Total Connections: {total_connections}")
            print(f"Average Connect Time: {avg_connect_time:.1f}ms")
            
            # Response stats
            response_stats = results.get("response_stats", {})
            success_rate = response_stats.get("success_rate", 0)
            avg_response_time = response_stats.get("average_response_time_ms", 0)
            print(f"Message Success Rate: {success_rate:.1f}%")
            print(f"Average Response Time: {avg_response_time:.1f}ms")
            
            # Pipeline stats
            pipeline_stats = results.get("pipeline_stats", {})
            avg_e2e = pipeline_stats.get("average_end_to_end_ms", 0)
            if avg_e2e > 0:
                print(f"Average End-to-End: {avg_e2e:.1f}ms")
        
        print("="*60)

async def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="WebSocket Voice Pipeline Diagnostics")
    parser.add_argument("--url", default="http://localhost:8080", help="Base URL for the voice agent")
    parser.add_argument("--test", choices=["quick", "comprehensive", "monitor"], default="quick",
                       help="Type of test to run")
    parser.add_argument("--duration", type=int, default=300, help="Duration for monitoring (seconds)")
    parser.add_argument("--output-dir", default="diagnostic_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create diagnostic runner
    runner = DiagnosticRunner(args.url)
    runner.results_dir = Path(args.output_dir)
    runner.results_dir.mkdir(exist_ok=True)
    
    try:
        if args.test == "quick":
            results = await runner.run_quick_check()
            filename = runner.save_results(results, "quick_check")
            runner.print_summary(results)
            
        elif args.test == "comprehensive":
            results = await runner.run_comprehensive_diagnostics()
            filename = runner.save_results(results, "comprehensive_diagnostics")
            runner.print_summary(results)
            
        elif args.test == "monitor":
            results = await runner.run_monitoring_session(args.duration)
            filename = runner.save_results(results, "monitoring_session")
            runner.print_summary(results)
        
        print(f"\n📄 Detailed results saved to: {filename}")
        
    except KeyboardInterrupt:
        logger.info("🛑 Diagnostics interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Diagnostics failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Check for required packages
    missing_packages = []
    
    try:
        import numpy
    except ImportError:
        missing_packages.append("numpy")
    
    try:
        import websockets
    except ImportError:
        missing_packages.append("websockets")
    
    try:
        import psutil
    except ImportError:
        missing_packages.append("psutil")
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print(f"Please install with: pip install {' '.join(missing_packages)}")
        sys.exit(1)
    
    asyncio.run(main())