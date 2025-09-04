#!/usr/bin/env python3
"""
🧠 MELVIN BRAIN MONITOR - Real-time Neural Activity Tracking
===========================================================
Monitor Melvin's brain activity, node creation, connection formation, and learning patterns.
Track Hebbian learning, pruning decisions, and neural network growth in real-time.
"""

import time
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque

from melvin_optimized_v2 import MelvinOptimizedV2, OptimizedContinuousFeeder, ContentType, ConnectionType

class MelvinBrainMonitor:
    """Real-time monitoring system for Melvin's brain activity"""
    
    def __init__(self, melvin_brain: MelvinOptimizedV2):
        self.melvin = melvin_brain
        self.monitoring = False
        self.monitor_thread = None
        
        # Activity tracking
        self.activity_log = deque(maxlen=10000)
        self.node_creation_times = []
        self.connection_formation_times = []
        self.hebbian_events = []
        self.pruning_events = []
        
        # Real-time statistics
        self.stats_history = deque(maxlen=1000)
        self.connection_types = defaultdict(int)
        self.content_types = defaultdict(int)
        self.activation_patterns = defaultdict(int)
        
        # Learning analysis
        self.learning_rate = []
        self.importance_scores = []
        self.compression_ratios = []
        
        # Monitoring configuration
        self.update_interval = 1.0  # seconds
        self.detailed_logging = True
        
        print("🧠 Melvin Brain Monitor initialized")
    
    def start_monitoring(self):
        """Start real-time brain monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("📊 Brain monitoring started - tracking neural activity...")
    
    def stop_monitoring(self):
        """Stop brain monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("📊 Brain monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Capture current brain state
                state = self.melvin.get_unified_state()
                self._record_activity(state)
                
                # Update statistics
                self._update_stats(state)
                
                # Log detailed activity
                if self.detailed_logging:
                    self._log_detailed_activity(state)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                time.sleep(self.update_interval)
    
    def _record_activity(self, state: Dict[str, Any]):
        """Record brain activity"""
        timestamp = time.time()
        
        activity = {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp).isoformat(),
            'nodes': state['global_memory']['total_nodes'],
            'connections': state['global_memory']['total_edges'],
            'storage_mb': state['global_memory']['storage_used_mb'],
            'hebbian_updates': state['global_memory']['stats']['hebbian_updates'],
            'uptime_seconds': state['system']['uptime_seconds']
        }
        
        self.activity_log.append(activity)
        self.stats_history.append(activity)
    
    def _update_stats(self, state: Dict[str, Any]):
        """Update real-time statistics"""
        # Track connection types
        if hasattr(self.melvin.binary_storage, 'total_connections'):
            # This would need to be enhanced to track actual connection types
            pass
        
        # Track content types
        if hasattr(self.melvin.binary_storage, 'total_nodes'):
            # This would need to be enhanced to track actual content types
            pass
    
    def _log_detailed_activity(self, state: Dict[str, Any]):
        """Log detailed brain activity"""
        current_time = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n🧠 [{current_time}] BRAIN ACTIVITY UPDATE:")
        print(f"   📦 Nodes: {state['global_memory']['total_nodes']:,}")
        print(f"   🔗 Connections: {state['global_memory']['total_edges']:,}")
        print(f"   💾 Storage: {state['global_memory']['storage_used_mb']:.2f}MB")
        print(f"   ⚡ Hebbian Updates: {state['global_memory']['stats']['hebbian_updates']}")
        print(f"   🕐 Uptime: {state['system']['uptime_seconds']:.1f}s")
        
        # Calculate growth rates
        if len(self.stats_history) > 1:
            prev = self.stats_history[-2]
            curr = self.stats_history[-1]
            
            node_growth = curr['nodes'] - prev['nodes']
            connection_growth = curr['connections'] - prev['connections']
            
            if node_growth > 0:
                print(f"   📈 Node Growth: +{node_growth}")
            if connection_growth > 0:
                print(f"   🔗 Connection Growth: +{connection_growth}")
    
    def feed_data_and_monitor(self, data_sources: List[str], duration: Optional[int] = None):
        """Feed data to Melvin while monitoring brain activity"""
        print(f"🚀 Starting data feeding with brain monitoring...")
        print(f"📖 Data sources: {len(data_sources)}")
        print(f"⏱️ Duration: {'Continuous' if duration is None else f'{duration}s'}")
        
        # Start monitoring
        self.start_monitoring()
        
        # Create feeder
        feeder = OptimizedContinuousFeeder(max_storage_gb=4000)
        
        start_time = time.time()
        try:
            # Process data sources
            for i, source_path in enumerate(data_sources):
                if duration and (time.time() - start_time) > duration:
                    print(f"⏰ Time limit reached ({duration}s)")
                    break
                
                print(f"\n📖 Processing source {i+1}/{len(data_sources)}: {source_path}")
                
                # Extract and process content
                content_list = feeder._extract_content_from_source(source_path)
                
                if not content_list:
                    print(f"⚠️ No content extracted from {source_path}")
                    continue
                
                # Process each content item with monitoring
                for j, content in enumerate(content_list):
                    if duration and (time.time() - start_time) > duration:
                        break
                    
                    if isinstance(content, str):
                        # Process text input
                        node_id = self.melvin.process_text_input(content, "monitor_feeder")
                        
                        # Record node creation
                        self.node_creation_times.append({
                            'timestamp': time.time(),
                            'node_id': node_id.hex()[:8],
                            'content_type': 'TEXT',
                            'content_preview': content[:50] + "..." if len(content) > 50 else content
                        })
                        
                        # Monitor Hebbian learning
                        if hasattr(self.melvin, 'recent_activations'):
                            recent_count = len(self.melvin.recent_activations)
                            if recent_count > 1:
                                self.hebbian_events.append({
                                    'timestamp': time.time(),
                                    'node_id': node_id.hex()[:8],
                                    'co_activations': recent_count - 1
                                })
                    
                    # Progress update
                    if j % 10 == 0:
                        state = self.melvin.get_unified_state()
                        print(f"   📊 Progress: {j+1}/{len(content_list)} items, "
                              f"{state['global_memory']['total_nodes']} nodes, "
                              f"{state['global_memory']['total_edges']} connections")
                
                print(f"✅ Completed {source_path}")
                
        except KeyboardInterrupt:
            print("\n🛑 Data feeding interrupted by user")
        finally:
            # Stop monitoring
            self.stop_monitoring()
            
            # Final report
            self._generate_final_report()
    
    def _generate_final_report(self):
        """Generate comprehensive brain activity report"""
        print("\n" + "="*60)
        print("🧠 MELVIN BRAIN ACTIVITY REPORT")
        print("="*60)
        
        if not self.stats_history:
            print("❌ No activity data recorded")
            return
        
        # Calculate statistics
        initial_state = self.stats_history[0]
        final_state = self.stats_history[-1]
        
        # Growth statistics
        total_nodes_created = final_state['nodes'] - initial_state['nodes']
        total_connections_formed = final_state['connections'] - initial_state['connections']
        total_hebbian_updates = final_state['hebbian_updates'] - initial_state['hebbian_updates']
        
        # Time statistics
        total_time = final_state['uptime_seconds'] - initial_state['uptime_seconds']
        
        # Rates
        nodes_per_second = total_nodes_created / total_time if total_time > 0 else 0
        connections_per_second = total_connections_formed / total_time if total_time > 0 else 0
        hebbian_per_second = total_hebbian_updates / total_time if total_time > 0 else 0
        
        print(f"\n📊 GROWTH STATISTICS:")
        print(f"   🧠 Nodes Created: {total_nodes_created:,}")
        print(f"   🔗 Connections Formed: {total_connections_formed:,}")
        print(f"   ⚡ Hebbian Updates: {total_hebbian_updates:,}")
        print(f"   💾 Storage Used: {final_state['storage_mb']:.2f}MB")
        
        print(f"\n⏱️ TIME STATISTICS:")
        print(f"   🕐 Total Time: {total_time:.1f}s")
        print(f"   📈 Nodes/Second: {nodes_per_second:.2f}")
        print(f"   🔗 Connections/Second: {connections_per_second:.2f}")
        print(f"   ⚡ Hebbian/Second: {hebbian_per_second:.2f}")
        
        print(f"\n🎯 LEARNING PATTERNS:")
        print(f"   🧠 Node Creation Events: {len(self.node_creation_times)}")
        print(f"   🔗 Hebbian Events: {len(self.hebbian_events)}")
        print(f"   📊 Activity Logs: {len(self.activity_log)}")
        
        # Connection formation analysis
        if self.hebbian_events:
            avg_co_activations = sum(e['co_activations'] for e in self.hebbian_events) / len(self.hebbian_events)
            print(f"   🔗 Avg Co-activations: {avg_co_activations:.2f}")
        
        # Storage efficiency
        if total_nodes_created > 0:
            avg_node_size = final_state['storage_mb'] * 1024 * 1024 / final_state['nodes']
            print(f"   💾 Avg Node Size: {avg_node_size:.1f} bytes")
        
        print(f"\n📈 FINAL STATE:")
        print(f"   🧠 Total Nodes: {final_state['nodes']:,}")
        print(f"   🔗 Total Connections: {final_state['connections']:,}")
        print(f"   💾 Total Storage: {final_state['storage_mb']:.2f}MB")
        print(f"   ⚡ Total Hebbian: {final_state['hebbian_updates']:,}")
        
        # Save detailed report
        self._save_detailed_report()
    
    def _save_detailed_report(self):
        """Save detailed brain activity report to file"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_nodes': self.stats_history[-1]['nodes'] if self.stats_history else 0,
                'total_connections': self.stats_history[-1]['connections'] if self.stats_history else 0,
                'total_storage_mb': self.stats_history[-1]['storage_mb'] if self.stats_history else 0,
                'total_hebbian': self.stats_history[-1]['hebbian_updates'] if self.stats_history else 0
            },
            'activity_log': list(self.activity_log),
            'node_creation_times': self.node_creation_times,
            'hebbian_events': self.hebbian_events,
            'stats_history': list(self.stats_history)
        }
        
        # Save to file
        report_file = f"melvin_brain_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get current real-time statistics"""
        if not self.stats_history:
            return {}
        
        current = self.stats_history[-1]
        
        # Calculate rates
        if len(self.stats_history) > 1:
            prev = self.stats_history[-2]
            time_diff = current['timestamp'] - prev['timestamp']
            
            node_rate = (current['nodes'] - prev['nodes']) / time_diff if time_diff > 0 else 0
            connection_rate = (current['connections'] - prev['connections']) / time_diff if time_diff > 0 else 0
        else:
            node_rate = 0
            connection_rate = 0
        
        return {
            'current_nodes': current['nodes'],
            'current_connections': current['connections'],
            'current_storage_mb': current['storage_mb'],
            'current_hebbian': current['hebbian_updates'],
            'node_rate_per_second': node_rate,
            'connection_rate_per_second': connection_rate,
            'uptime_seconds': current['uptime_seconds']
        }

def main():
    """Main function for brain monitoring demonstration"""
    print("🧠 MELVIN BRAIN MONITOR")
    print("=" * 50)
    
    # Initialize Melvin's optimized brain
    melvin = MelvinOptimizedV2()
    
    # Initialize brain monitor
    monitor = MelvinBrainMonitor(melvin)
    
    # Define data sources to feed
    data_sources = [
        "README.md",
        "melvin_global_brain.py",
        "melvin_optimized_v2.py",
        "MELVIN_OPTIMIZED_V2_SUMMARY.md"
    ]
    
    # Start feeding data with monitoring
    print("🚀 Starting brain activity monitoring...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        # Feed data and monitor brain activity
        monitor.feed_data_and_monitor(data_sources, duration=60)  # Monitor for 60 seconds
        
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")
        monitor.stop_monitoring()
        monitor._generate_final_report()
    
    print("\n🎉 Brain monitoring session completed!")

if __name__ == "__main__":
    main()
