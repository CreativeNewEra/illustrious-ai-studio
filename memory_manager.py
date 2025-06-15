#!/usr/bin/env python3
"""
Enhanced Memory Manager for Illustrious AI Studio
Comprehensive memory management and OOM prevention tool
"""

import argparse
import sys
import time

from core.memory_guardian import get_memory_guardian, MemoryPressureLevel
from core.state import AppState
from core.config import CONFIG
try:
    import torch
except Exception:  # pragma: no cover - allow missing torch in tests
    torch = None  # type: ignore

def create_parser():
    """Create argument parser for memory manager CLI"""
    parser = argparse.ArgumentParser(description="Memory Manager for Illustrious AI Studio")
    parser.add_argument("--status", action="store_true", help="Show current memory status")
    parser.add_argument("--monitor", action="store_true", help="Start interactive memory monitoring")
    parser.add_argument("--clear", action="store_true", help="Clear GPU memory cache")
    parser.add_argument("--report", action="store_true", help="Generate memory usage report")
    parser.add_argument("--test-pressure", choices=['medium', 'high', 'critical'], 
                       help="Test memory pressure handling")
    parser.add_argument("--config", action="store_true", help="Show memory guardian configuration")
    parser.add_argument("--threshold", action="append", metavar="LEVEL:PERCENT",
                       help="Set memory threshold (e.g. medium:80)")
    parser.add_argument(
        "--profile",
        choices=["conservative", "balanced", "aggressive"],
        help="Set memory guardian profile",
    )
    parser.add_argument("--enable", action="store_true", help="Enable memory guardian")
    parser.add_argument("--disable", action="store_true", help="Disable memory guardian")
    return parser

class MemoryManagerCLI:
    def __init__(self):
        self.app_state = AppState()
        self.guardian = get_memory_guardian(self.app_state)
        
    def show_status(self):
        """Show current memory status"""
        print("🛡️ Memory Guardian Status")
        print("=" * 50)
        
        stats = self.guardian.get_memory_stats()
        if not stats:
            print("❌ Unable to get memory statistics (GPU not available)")
            return
            
        # Current status
        print(f"Guardian Status: {'🟢 Active' if self.guardian.is_monitoring else '🔴 Inactive'}")
        print(f"GPU Total:       {stats.gpu_total_gb:.1f} GB")
        print(f"GPU Used:        {stats.gpu_reserved_gb:.1f} GB ({stats.gpu_usage_percent:.1f}%)")
        print(f"GPU Free:        {stats.gpu_free_gb:.1f} GB")
        print(f"System RAM:      {stats.system_ram_usage_percent:.1f}% used")
        
        # Pressure level with color coding
        pressure_colors = {
            "low": "🟢",
            "medium": "🟡", 
            "high": "🟠",
            "critical": "🔴"
        }
        color = pressure_colors.get(stats.pressure_level.value, "⚪")
        print(f"Pressure Level:  {color} {stats.pressure_level.value.upper()}")
        
        # Intervention stats
        print(f"\nInterventions:   {self.guardian.interventions_count}")
        print(f"OOMs Prevented:  {self.guardian.oom_prevented_count}")
        
    def show_report(self):
        """Generate comprehensive memory report"""
        print("📊 Memory Usage Report")
        print("=" * 50)
        
        report = self.guardian.get_memory_report()
        
        print(f"Guardian Status: {report['guardian_status']}")
        print(f"Total Interventions: {report['interventions_count']}")
        print(f"OOMs Prevented: {report['oom_prevented_count']}")
        
        if report.get('current_stats'):
            stats = report['current_stats']
            print(f"\nCurrent Memory:")
            print(f"  GPU: {stats['gpu_usage_percent']:.1f}% ({stats['gpu_free_gb']:.1f}GB free)")
            print(f"  RAM: {stats['system_ram_usage_percent']:.1f}%")
            print(f"  Pressure: {stats['pressure_level']}")
        
        print(f"\nThresholds:")
        for level, percent in report['thresholds'].items():
            print(f"  {level.capitalize()}: {percent}")
            
        if report.get('recent_usage'):
            print(f"\nRecent Usage (last 10 readings):")
            for usage in report['recent_usage'][-5:]:  # Show last 5
                timestamp = usage['timestamp'].split('T')[1][:8]  # Just time
                print(f"  {timestamp}: {usage['gpu_usage']} ({usage['pressure']})")
    
    def show_config(self):
        """Show memory guardian configuration"""
        print("⚙️ Memory Guardian Configuration")
        print("=" * 50)
        
        config = self.guardian.config
        for key, value in config.items():
            print(f"{key:25}: {value}")
            
        print(f"\nThresholds:")
        thresholds = self.guardian.thresholds
        print(f"Low (monitoring):    {thresholds.low_threshold * 100:.0f}%")
        print(f"Medium (prevention): {thresholds.medium_threshold * 100:.0f}%")
        print(f"High (aggressive):   {thresholds.high_threshold * 100:.0f}%")
        print(f"Critical (emergency):{thresholds.critical_threshold * 100:.0f}%")
        
        print(f"\nSafety Margins:")
        print(f"Image Generation:    {thresholds.generation_reserve_gb:.1f} GB")
        print(f"LLM Operations:      {thresholds.llm_reserve_gb:.1f} GB")
    
    def clear_memory(self):
        """Clear GPU memory"""
        print("🧹 Clearing GPU memory...")
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print("✅ GPU cache cleared")
            else:
                print("⚠️ CUDA not available")
                
            import gc
            collected = gc.collect()
            print(f"✅ Garbage collection freed {collected} objects")
            
        except Exception as e:
            print(f"❌ Error clearing memory: {e}")
    
    def monitor_interactive(self):
        """Start interactive memory monitoring"""
        print("📈 Interactive Memory Monitor")
        print("Press Ctrl+C to stop")
        print("=" * 50)
        
        try:
            while True:
                stats = self.guardian.get_memory_stats()
                if stats:
                    pressure_indicator = {
                        "low": "🟢",
                        "medium": "🟡",
                        "high": "🟠", 
                        "critical": "🔴"
                    }.get(stats.pressure_level.value, "⚪")
                    
                    print(f"\r{pressure_indicator} GPU: {stats.gpu_usage_percent:5.1f}% "
                          f"({stats.gpu_free_gb:4.1f}GB free) | "
                          f"RAM: {stats.system_ram_usage_percent:5.1f}% | "
                          f"Pressure: {stats.pressure_level.value:8s}", end="", flush=True)
                else:
                    print("\r❌ Unable to read memory stats", end="", flush=True)
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped")
    
    def test_pressure_handling(self, level: str):
        """Test memory pressure handling"""
        print(f"🧪 Testing {level} pressure handling...")
        
        # Create artificial pressure level
        pressure_map = {
            'medium': MemoryPressureLevel.MEDIUM,
            'high': MemoryPressureLevel.HIGH,
            'critical': MemoryPressureLevel.CRITICAL
        }
        
        pressure_level = pressure_map[level]
        print(f"Triggering {pressure_level.value} pressure interventions...")
        
        # Get current stats for reference
        stats = self.guardian.get_memory_stats()
        if stats:
            print(f"Before: {stats.gpu_usage_percent:.1f}% GPU usage")
        
        # Manually trigger interventions
        interventions = self.guardian.intervention_callbacks.get(pressure_level, [])
        print(f"Running {len(interventions)} interventions...")
        
        for intervention in interventions:
            try:
                result = intervention()
                status = "✅ Success" if result else "❌ Failed"
                print(f"  {intervention.__name__}: {status}")
            except Exception as e:
                print(f"  {intervention.__name__}: ❌ Error - {e}")
        
        # Check stats after
        time.sleep(1)
        stats_after = self.guardian.get_memory_stats()
        if stats_after:
            print(f"After:  {stats_after.gpu_usage_percent:.1f}% GPU usage")
    
    def set_threshold(self, threshold_spec: str):
        """Set memory threshold"""
        try:
            level, percent_str = threshold_spec.split(':')
            percent = float(percent_str)
            try:
                self.guardian.set_threshold(level, percent)
            except ValueError as ve:
                print(f"❌ {ve}")
                return

            print(f"✅ Set {level} threshold to {percent*100:.0f}%")
            
        except ValueError:
            print("❌ Invalid threshold format. Use LEVEL:PERCENT (e.g. medium:80)")
    
    def enable_guardian(self):
        """Enable memory guardian"""
        if not self.guardian.is_monitoring:
            self.guardian.start_monitoring()
            print("✅ Memory Guardian enabled")
        else:
            print("⚠️ Memory Guardian already running")
    
    def disable_guardian(self):
        """Disable memory guardian"""
        if self.guardian.is_monitoring:
            self.guardian.stop_monitoring()
            print("✅ Memory Guardian disabled")
        else:
            print("⚠️ Memory Guardian already stopped")

    def set_profile(self, profile: str):
        """Change memory guardian profile"""
        try:
            self.guardian.set_profile(profile)
            print(f"✅ Profile set to {profile}")
        except ValueError as ve:
            print(f"❌ {ve}")

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    cli = MemoryManagerCLI()
    
    try:
        if args.status:
            cli.show_status()
        elif args.report:
            cli.show_report()
        elif args.config:
            cli.show_config()
        elif args.clear:
            cli.clear_memory()
        elif args.monitor:
            cli.monitor_interactive()
        elif args.test_pressure:
            cli.test_pressure_handling(args.test_pressure)
        elif args.profile:
            cli.set_profile(args.profile)
        elif args.threshold:
            for th in args.threshold:
                cli.set_threshold(th)
        elif args.enable:
            cli.enable_guardian()
        elif args.disable:
            cli.disable_guardian()
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
