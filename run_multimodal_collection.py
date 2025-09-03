#!/usr/bin/env python3
"""
🚀 MELVIN MULTIMODAL COLLECTION RUNNER
=====================================
Comprehensive runner script for collecting multimodal datasets and integrating
them into Melvin's global memory system.

This script:
1. Initializes Melvin's global brain
2. Collects datasets from multiple modalities (visual, text, code, audio)
3. Converts data into node-connection format
4. Saves everything to Melvin's global memory
5. Provides detailed reporting and monitoring

Usage:
    python3 run_multimodal_collection.py [options]
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from melvin_global_brain import MelvinGlobalBrain, NodeType
    from melvin_multimodal_collector import MelvinMultimodalCollector, DatasetConfig
    MELVIN_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importing Melvin modules: {e}")
    MELVIN_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('melvin_collection.log')
    ]
)
logger = logging.getLogger(__name__)

class MelvinCollectionRunner:
    """Main runner for multimodal data collection"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config = self._load_config()
        self.global_brain = None
        self.collector = None
        
        # Runtime statistics
        self.start_time = time.time()
        self.session_stats = {
            'total_runtime': 0,
            'datasets_processed': 0,
            'total_samples': 0,
            'nodes_created': 0,
            'connections_created': 0,
            'errors_encountered': 0
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'brain_settings': {
                'embedding_dim': 512,
                'memory_path': 'melvin_global_memory',
                'enable_hebbian_learning': True,
                'hebbian_learning_rate': 0.01,
                'coactivation_window': 2.0
            },
            'collection_settings': {
                'output_dir': 'melvin_datasets',
                'max_samples_per_dataset': 50,
                'enable_cross_modal_connections': True,
                'save_intermediate_results': True,
                'parallel_processing': False
            },
            'dataset_configs': [
                {
                    'name': 'squad',
                    'source': 'huggingface',
                    'data_type': 'text',
                    'max_samples': 30,
                    'enabled': True
                },
                {
                    'name': 'imdb',
                    'source': 'huggingface',
                    'data_type': 'text',
                    'max_samples': 50,
                    'enabled': True
                },
                {
                    'name': 'code_examples',
                    'source': 'generated',
                    'data_type': 'code',
                    'max_samples': 40,
                    'enabled': True
                },
                {
                    'name': 'visual_features',
                    'source': 'generated',
                    'data_type': 'visual',
                    'max_samples': 60,
                    'enabled': True
                },
                {
                    'name': 'audio_features',
                    'source': 'generated',
                    'data_type': 'audio',
                    'max_samples': 30,
                    'enabled': True
                }
            ]
        }
        
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults
                default_config.update(loaded_config)
                logger.info(f"📄 Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"⚠️ Error loading config file: {e}, using defaults")
        
        return default_config
    
    def initialize_brain(self) -> bool:
        """Initialize Melvin's global brain"""
        try:
            if not MELVIN_AVAILABLE:
                logger.error("❌ Melvin modules not available")
                return False
            
            logger.info("🧠 Initializing Melvin Global Brain...")
            
            brain_settings = self.config['brain_settings']
            self.global_brain = MelvinGlobalBrain(
                embedding_dim=brain_settings['embedding_dim']
            )
            
            # Start unified processing for Hebbian learning
            self.global_brain.start_unified_processing()
            
            # Get initial state
            initial_state = self.global_brain.get_unified_state()
            logger.info(f"🧠 Brain initialized with {initial_state['global_memory']['total_nodes']} nodes, "
                       f"{initial_state['global_memory']['total_edges']} edges")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing brain: {e}")
            return False
    
    def initialize_collector(self) -> bool:
        """Initialize multimodal data collector"""
        try:
            if not self.global_brain:
                logger.error("❌ Global brain not initialized")
                return False
            
            logger.info("🤖 Initializing Multimodal Collector...")
            
            collection_settings = self.config['collection_settings']
            self.collector = MelvinMultimodalCollector(
                self.global_brain,
                output_dir=collection_settings['output_dir']
            )
            
            logger.info("✅ Multimodal collector initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing collector: {e}")
            return False
    
    def run_collection(self) -> Dict[str, Any]:
        """Run the complete multimodal data collection"""
        logger.info("🚀 Starting multimodal data collection...")
        
        try:
            # Convert config to DatasetConfig objects
            dataset_configs = []
            for config_dict in self.config['dataset_configs']:
                if config_dict.get('enabled', True):
                    dataset_config = DatasetConfig(
                        name=config_dict['name'],
                        source=config_dict['source'],
                        data_type=config_dict['data_type'],
                        max_samples=config_dict['max_samples'],
                        enabled=config_dict.get('enabled', True),
                        metadata=config_dict.get('metadata', {})
                    )
                    dataset_configs.append(dataset_config)
            
            logger.info(f"📊 Processing {len(dataset_configs)} datasets...")
            
            # Run collection
            results = self.collector.collect_all_datasets(dataset_configs)
            
            # Update session stats
            self.session_stats.update({
                'datasets_processed': results['collection_summary']['total_datasets'],
                'total_samples': results['collection_summary']['total_samples'],
                'nodes_created': results['collection_summary']['nodes_created'],
                'connections_created': results['collection_summary']['connections_created']
            })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error during collection: {e}")
            self.session_stats['errors_encountered'] += 1
            raise
    
    def save_to_global_memory(self) -> bool:
        """Save all collected data to Melvin's global memory"""
        try:
            logger.info("💾 Saving to Melvin's global memory...")
            
            # Save complete brain state
            self.global_brain.save_complete_state()
            
            # Create collection metadata
            metadata = {
                'collection_timestamp': time.time(),
                'session_stats': self.session_stats,
                'config_used': self.config,
                'brain_state_snapshot': self.global_brain.get_unified_state()
            }
            
            # Save metadata
            metadata_file = Path(self.config['collection_settings']['output_dir']) / "collection_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"✅ Data saved to global memory and {metadata_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving to global memory: {e}")
            return False
    
    def generate_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive collection report"""
        runtime = time.time() - self.start_time
        self.session_stats['total_runtime'] = runtime
        
        # Get final brain state
        final_brain_state = self.global_brain.get_unified_state()
        
        report = {
            'session_summary': {
                'start_time': self.start_time,
                'end_time': time.time(),
                'total_runtime': runtime,
                'runtime_formatted': f"{runtime:.2f} seconds"
            },
            'collection_results': results,
            'session_statistics': self.session_stats,
            'brain_growth': {
                'final_nodes': final_brain_state['global_memory']['total_nodes'],
                'final_edges': final_brain_state['global_memory']['total_edges'],
                'node_types': final_brain_state['global_memory']['node_types'],
                'edge_types': final_brain_state['global_memory']['edge_types'],
                'hebbian_updates': final_brain_state['global_memory']['stats'].get('hebbian_updates', 0)
            },
            'performance_metrics': {
                'samples_per_second': self.session_stats['total_samples'] / runtime if runtime > 0 else 0,
                'nodes_per_second': self.session_stats['nodes_created'] / runtime if runtime > 0 else 0,
                'processing_efficiency': 'high' if self.session_stats['errors_encountered'] == 0 else 'medium'
            },
            'recommendations': self._generate_recommendations(results, final_brain_state)
        }
        
        return report
    
    def _generate_recommendations(self, results: Dict[str, Any], brain_state: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on collection results"""
        recommendations = []
        
        # Check data balance
        modality_counts = results.get('modality_breakdown', {})
        total_samples = sum(modality_counts.values())
        
        if total_samples == 0:
            recommendations.append("⚠️ No data collected - check dataset configurations and HuggingFace availability")
        else:
            # Check modality balance
            for modality, count in modality_counts.items():
                ratio = count / total_samples
                if ratio < 0.1:
                    recommendations.append(f"📊 Consider increasing {modality} data samples (currently {ratio:.1%})")
                elif ratio > 0.6:
                    recommendations.append(f"⚖️ {modality} data dominates (currently {ratio:.1%}) - consider balancing")
        
        # Check connections
        total_edges = brain_state.get('total_edges', 0)
        total_nodes = brain_state.get('total_nodes', 0)
        if total_nodes > 0:
            connectivity = total_edges / total_nodes
            if connectivity < 2.0:
                recommendations.append("🔗 Low connectivity - consider enabling more cross-modal connections")
            elif connectivity > 50.0:
                recommendations.append("🔗 Very high connectivity - consider pruning weak connections")
        
        # Check Hebbian learning
        hebbian_updates = brain_state.get('hebbian_updates', 0)
        if hebbian_updates == 0:
            recommendations.append("🧠 No Hebbian learning detected - ensure background processing is enabled")
        
        # Performance recommendations
        if self.session_stats['errors_encountered'] > 0:
            recommendations.append("❌ Errors encountered - check logs and dataset availability")
        
        if not recommendations:
            recommendations.append("✅ Collection completed successfully with balanced multimodal data")
        
        return recommendations
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete collection pipeline"""
        try:
            print("🤖 MELVIN MULTIMODAL COLLECTION PIPELINE")
            print("=" * 60)
            print("🔹 INITIALIZING GLOBAL BRAIN")
            print("🔹 COLLECTING MULTIMODAL DATASETS")
            print("🔹 CREATING NODE-CONNECTION NETWORKS")
            print("🔹 ENABLING HEBBIAN LEARNING")
            print("🔹 SAVING TO GLOBAL MEMORY")
            print("=" * 60)
            
            # Step 1: Initialize brain
            if not self.initialize_brain():
                return False
            
            # Step 2: Initialize collector
            if not self.initialize_collector():
                return False
            
            # Step 3: Run collection
            results = self.run_collection()
            
            # Step 4: Save to global memory
            if not self.save_to_global_memory():
                return False
            
            # Step 5: Generate and display report
            report = self.generate_report(results)
            self.display_report(report)
            
            # Save final report
            report_file = Path(self.config['collection_settings']['output_dir']) / "final_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📊 Final report saved to {report_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline error: {e}")
            return False
        finally:
            self.cleanup()
    
    def display_report(self, report: Dict[str, Any]):
        """Display formatted collection report"""
        print("\n🎉 COLLECTION COMPLETE!")
        print("=" * 50)
        
        # Session summary
        session = report['session_summary']
        print(f"⏱️ Runtime: {session['runtime_formatted']}")
        
        # Collection results
        collection = report['collection_results']['collection_summary']
        print(f"📊 Datasets processed: {collection['total_datasets']}")
        print(f"📦 Total samples: {collection['total_samples']}")
        print(f"🧠 Nodes created: {collection['nodes_created']}")
        print(f"🔗 Connections: {collection['connections_created']}")
        
        # Modality breakdown
        print(f"\n🎯 MODALITY BREAKDOWN:")
        modality_breakdown = report['collection_results']['modality_breakdown']
        for modality, count in modality_breakdown.items():
            percentage = (count / collection['total_samples'] * 100) if collection['total_samples'] > 0 else 0
            print(f"   {modality}: {count} samples ({percentage:.1f}%)")
        
        # Brain state
        print(f"\n🧠 FINAL BRAIN STATE:")
        brain = report['brain_growth']
        print(f"   Total nodes: {brain['final_nodes']}")
        print(f"   Total edges: {brain['final_edges']}")
        print(f"   Cross-modal connections: {brain['edge_types'].get('multimodal', 0)}")
        print(f"   Hebbian updates: {brain['hebbian_updates']}")
        
        # Performance
        print(f"\n⚡ PERFORMANCE:")
        perf = report['performance_metrics']
        print(f"   Samples/sec: {perf['samples_per_second']:.1f}")
        print(f"   Nodes/sec: {perf['nodes_per_second']:.1f}")
        print(f"   Efficiency: {perf['processing_efficiency']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report['recommendations']:
            print(f"   {rec}")
        
        print("\n💾 All data saved to Melvin's global memory!")
        print("🧠 Hebbian learning active - connections will strengthen over time")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.global_brain:
                self.global_brain.stop_unified_processing()
                logger.info("🧠 Global brain processing stopped")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Melvin Multimodal Data Collection Runner")
    parser.add_argument("--config", "-c", help="Configuration file path", default=None)
    parser.add_argument("--output-dir", "-o", help="Output directory", default="melvin_datasets")
    parser.add_argument("--max-samples", "-m", type=int, help="Max samples per dataset", default=50)
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create runner
        runner = MelvinCollectionRunner(config_file=args.config)
        
        # Override config with command line args
        if args.output_dir:
            runner.config['collection_settings']['output_dir'] = args.output_dir
        if args.max_samples:
            for dataset_config in runner.config['dataset_configs']:
                dataset_config['max_samples'] = min(dataset_config['max_samples'], args.max_samples)
        
        # Run complete pipeline
        success = runner.run_complete_pipeline()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n🛑 Collection interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
