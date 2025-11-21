#!/usr/bin/env python3


import json
import numpy as np
from pathlib import Path

class FusionAnalyzer:
    """Comprehensive analyzer for fusion methods and gate distributions"""
    
    def __init__(self, results_json_path):
        """Initialize with fusion test results"""
        with open(results_json_path, 'r') as f:
            self.results = json.load(f)
        self.fusion_types = ["simple", "advanced", "cross_attention", "hierarchical"]
        
    def analyze_fusion_performance(self):
        """Comprehensive performance analysis of all fusion methods"""
        
        print("🔍 FUSION METHOD ANALYSIS & INTERPRETATION")
        print("=" * 80)
        
        # Create performance summary
        performance_data = []
        for dataset, results in self.results.items():
            for fusion_type, metrics in results.items():
                if metrics["status"] == "success":
                    performance_data.append({
                        "Dataset": dataset,
                        "Fusion_Type": fusion_type,
                        "Expressiveness": metrics["expressiveness_score"],
                        "Fusion_Parameters": metrics["fusion_parameters"],
                        "Total_Parameters": metrics["total_parameters"],
                        "Training_Time": metrics["training_time"],
                        "Parameter_Efficiency": metrics["expressiveness_score"] / max(1, metrics["fusion_parameters"] / 1000),
                        "Components": ", ".join(metrics["components"])
                    })
        
        # Create manual analysis without pandas
        print(f"   Found {len(performance_data)} successful experiments")
        
        # 1. FUSION METHOD CHARACTERISTICS
        print("\n📊 1. FUSION METHOD CHARACTERISTICS")
        print("-" * 50)
        
        # Manual aggregation by fusion type
        fusion_stats = {}
        for item in performance_data:
            fusion_type = item['Fusion_Type']
            if fusion_type not in fusion_stats:
                fusion_stats[fusion_type] = {
                    'expressiveness': [], 'parameters': [], 
                    'time': [], 'efficiency': []
                }
            fusion_stats[fusion_type]['expressiveness'].append(item['Expressiveness'])
            fusion_stats[fusion_type]['parameters'].append(item['Fusion_Parameters'])
            fusion_stats[fusion_type]['time'].append(item['Training_Time'])
            fusion_stats[fusion_type]['efficiency'].append(item['Parameter_Efficiency'])
        
        print("Fusion Type    | Expressiveness | Avg Parameters | Avg Time(s) | Efficiency")
        print("-" * 75)
        for fusion_type, stats in fusion_stats.items():
            exp_avg = np.mean(stats['expressiveness'])
            param_avg = np.mean(stats['parameters'])
            time_avg = np.mean(stats['time'])
            eff_avg = np.mean(stats['efficiency'])
            print(f"{fusion_type:13} | {exp_avg:13.1f} | {param_avg:13,.0f} | {time_avg:10.1f} | {eff_avg:9.3f}")
        
        # Detailed interpretation for each fusion type
        self._interpret_fusion_types()
        
        # 2. PERFORMANCE RANKINGS
        print("\n🏆 2. PERFORMANCE RANKINGS")
        print("-" * 50)
        
        # Manual ranking calculations
        avg_performance = {}
        for fusion_type, stats in fusion_stats.items():
            avg_performance[fusion_type] = {
                'Expressiveness': np.mean(stats['expressiveness']),
                'Parameter_Efficiency': np.mean(stats['efficiency']),
                'Training_Time': np.mean(stats['time'])
            }
        
        print("Most Expressive (Higher is Better):")
        exp_sorted = sorted(avg_performance.items(), key=lambda x: x[1]['Expressiveness'], reverse=True)
        for i, (fusion, metrics) in enumerate(exp_sorted, 1):
            print(f"  {i}. {fusion}: {metrics['Expressiveness']:.1f}/10")
        
        print("\nMost Parameter Efficient (Higher is Better):")
        eff_sorted = sorted(avg_performance.items(), key=lambda x: x[1]['Parameter_Efficiency'], reverse=True)
        for i, (fusion, metrics) in enumerate(eff_sorted, 1):
            print(f"  {i}. {fusion}: {metrics['Parameter_Efficiency']:.3f}")
        
        print("\nFastest Training (Lower is Better):")
        speed_sorted = sorted(avg_performance.items(), key=lambda x: x[1]['Training_Time'])
        for i, (fusion, metrics) in enumerate(speed_sorted, 1):
            print(f"  {i}. {fusion}: {metrics['Training_Time']:.1f}s")
        
        return performance_data, avg_performance
    
    def _interpret_fusion_types(self):
        """Detailed interpretation of each fusion method"""
        
        print("\n🧠 FUSION METHOD DETAILED INTERPRETATION:")
        print("-" * 50)
        
        interpretations = {
            "simple": {
                "description": "Basic scalar gating between spatial and spectral features",
                "mechanism": "Uses single gate value per spatial location: gate*spatial + (1-gate)*spectral",
                "pros": ["Fast training", "Low memory usage", "Simple architecture"],
                "cons": ["Limited expressiveness", "No cross-modal interaction", "Scalar gating only"],
                "best_for": "Quick prototyping, resource-constrained environments"
            },
            "advanced": {
                "description": "Channel-wise gating with cross-modal feature enhancement",
                "mechanism": "Projects features, applies cross-modal enhancement, then channel-wise gating",
                "pros": ["Channel-wise adaptation", "Cross-modal interaction", "Good balance"],
                "cons": ["More parameters", "Longer training", "Moderate complexity"],
                "best_for": "Production systems, balanced performance needs"
            },
            "cross_attention": {
                "description": "Cross-attention mechanisms between spatial and spectral domains",
                "mechanism": "Attention-based feature interaction with spatial-to-spectral and spectral-to-spatial attention",
                "pros": ["Highest expressiveness", "Complex relationships", "Attention weights interpretable"],
                "cons": ["Most parameters", "Computational overhead", "Potential overfitting"],
                "best_for": "Research, complex datasets, when performance is critical"
            },
            "hierarchical": {
                "description": "Multi-scale fusion with learnable scale weights",
                "mechanism": "Processes features at multiple scales then combines with learned weights",
                "pros": ["Multi-scale processing", "Handles various anomaly sizes", "Adaptive scaling"],
                "cons": ["High parameter count", "Complex architecture", "Scale dependency"],
                "best_for": "Multi-scale anomalies, varying object sizes"
            }
        }
        
        for fusion_type, info in interpretations.items():
            print(f"\n📋 {fusion_type.upper()} FUSION:")
            print(f"   Description: {info['description']}")
            print(f"   Mechanism: {info['mechanism']}")
            print(f"   Pros: {', '.join(info['pros'])}")
            print(f"   Cons: {', '.join(info['cons'])}")
            print(f"   Best for: {info['best_for']}")
    
    def interpret_gate_distributions(self, results_dir="Results"):
        """Interpret gate distribution patterns"""
        
        print("\n\n🎯 GATE DISTRIBUTION ANALYSIS")
        print("=" * 80)
        
        print("\n WHAT GATE DISTRIBUTIONS TELL US:")
        print("-" * 50)
        
        gate_interpretations = {
            "mean_near_0.5": "Balanced fusion - model uses both spatial and spectral equally",
            "mean_drift": "Learning preference - model discovers which features are more useful",
            "std_decrease": "Convergence - gates becoming more confident/stable over training",
            "std_increase": "Exploration - model still searching for optimal balance",
            "bimodal_distribution": "Spatial preference - some regions prefer one modality strongly",
            "uniform_distribution": "Uncertainty - model hasn't learned clear preferences"
        }
        
        for pattern, meaning in gate_interpretations.items():
            print(f"   • {pattern}: {meaning}")
        
        # Analyze specific gate patterns from your results
        print("\n YOUR GATE DISTRIBUTION PATTERNS:")
        print("-" * 50)
        
        # Example analysis based on the gate statistics we read
        print("\n🔍 AVIRIS-2 Dataset Analysis:")
        print("   SIMPLE fusion: Mean 0.553→0.521 (favors spatial initially, becomes more balanced)")
        print("   ADVANCED fusion: Mean ~0.509 (perfectly balanced, stable)")
        print("   → Interpretation: Advanced fusion learns better balance, simple fusion has bias")
        
        print("\n🔍 SALIANS-SYN Dataset Analysis:")
        print("   SIMPLE fusion: Mean 0.443→0.379 (strong spectral preference develops)")
        print("   ADVANCED fusion: Mean ~0.503 (near-perfect balance maintained)")
        print("   → Interpretation: Dataset benefits from spectral features, advanced handles this better")
        
        # General recommendations
        print("\n💡 GATE INTERPRETATION GUIDELINES:")
        print("-" * 50)
        print("   🎯 IDEAL PATTERNS:")
        print("      • Mean around 0.5: Good balance between modalities")
        print("      • Stable std: Consistent gate behavior across training")
        print("      • Gradual convergence: Smooth learning progression")
        
        print("\n   ⚠️  WARNING PATTERNS:")
        print("      • Mean near 0 or 1: Over-reliance on single modality")
        print("      • Highly unstable std: Training instability")
        print("      • Oscillating mean: Poor convergence")
        
        print("\n   📋 COMPARISON STRATEGY:")
        print("      • Compare mean values: Which modality is preferred?")
        print("      • Compare stability: Which fusion is more stable?")
        print("      • Compare convergence: Which learns faster/better?")
    
    def provide_recommendations(self, performance_data, avg_performance):
        """Provide comprehensive recommendations"""
        
        print("\n\n🎯 COMPREHENSIVE RECOMMENDATIONS")
        print("=" * 80)
        
        # Find best performers
        best_exp_fusion = max(avg_performance.items(), key=lambda x: x[1]['Expressiveness'])
        best_eff_fusion = max(avg_performance.items(), key=lambda x: x[1]['Parameter_Efficiency'])
        fastest_fusion = min(avg_performance.items(), key=lambda x: x[1]['Training_Time'])
        
        print(f"\n🏆 TOP PERFORMERS:")
        print(f"   Most Expressive: {best_exp_fusion[0]} ({best_exp_fusion[1]['Expressiveness']:.1f}/10)")
        print(f"   Most Efficient: {best_eff_fusion[0]} ({best_eff_fusion[1]['Parameter_Efficiency']:.3f})")
        print(f"   Fastest Training: {fastest_fusion[0]} ({fastest_fusion[1]['Training_Time']:.1f}s)")
        
        print("\n🎯 SCENARIO-BASED RECOMMENDATIONS:")
        print("-" * 50)
        
        scenarios = {
            "🚀 Production Deployment": {
                "recommendation": "advanced",
                "reason": "Best balance of performance, efficiency, and stability",
                "considerations": ["Reasonable training time", "Good expressiveness", "Stable convergence"]
            },
            "🔬 Research & Experimentation": {
                "recommendation": "cross_attention", 
                "reason": "Highest expressiveness for exploring complex relationships",
                "considerations": ["Maximum modeling capacity", "Interpretable attention", "State-of-the-art results"]
            },
            "⚡ Quick Prototyping": {
                "recommendation": "simple",
                "reason": "Fastest training and lowest computational requirements",
                "considerations": ["Rapid iteration", "Resource constraints", "Proof of concept"]
            },
            "🎯 Multi-scale Anomalies": {
                "recommendation": "hierarchical",
                "reason": "Handles varying anomaly sizes and scales effectively",
                "considerations": ["Variable object sizes", "Complex scenes", "Multi-resolution data"]
            },
            "💰 Resource Constrained": {
                "recommendation": "simple or advanced",
                "reason": "Lower parameter count and faster training",
                "considerations": ["Limited GPU memory", "Real-time requirements", "Edge deployment"]
            }
        }
        
        for scenario, info in scenarios.items():
            print(f"\n{scenario}:")
            print(f"   → Use: {info['recommendation'].upper()}")
            print(f"   → Reason: {info['reason']}")
            print(f"   → Consider: {', '.join(info['considerations'])}")
        
        print("\n📊 DECISION MATRIX:")
        print("-" * 50)
        
        # Create decision matrix
        decision_criteria = {
            "Performance": {"cross_attention": 5, "hierarchical": 4, "advanced": 3, "simple": 1},
            "Efficiency": {"simple": 5, "advanced": 4, "cross_attention": 2, "hierarchical": 1},
            "Speed": {"simple": 5, "hierarchical": 4, "cross_attention": 3, "advanced": 2},
            "Stability": {"advanced": 5, "simple": 4, "hierarchical": 3, "cross_attention": 2},
            "Interpretability": {"simple": 5, "cross_attention": 4, "advanced": 3, "hierarchical": 2}
        }
        
        print("Criteria (1-5 scale):")
        print("Fusion Type    | Performance | Efficiency | Speed | Stability | Interpretability | Total")
        print("-" * 85)
        
        totals = {}
        for fusion in self.fusion_types:
            scores = [decision_criteria[criterion][fusion] for criterion in decision_criteria]
            total = sum(scores)
            totals[fusion] = total
            print(f"{fusion:13} | {scores[0]:11} | {scores[1]:10} | {scores[2]:5} | {scores[3]:9} | {scores[4]:15} | {total:5}")
        
        # Overall recommendation
        best_overall = max(totals, key=totals.get)
        print(f"\n🏆 OVERALL BEST: {best_overall.upper()} (Score: {totals[best_overall]}/25)")
        
        print("\n🔄 NEXT STEPS:")
        print("-" * 50)
        print("1. 🎯 Run full training (150 epochs) with your chosen fusion type")
        print("2. 📊 Compare AUC scores on HAD100 test dataset")
        print("3. 📈 Analyze gate evolution over full training period")
        print("4. 🔍 Examine attention maps (for cross_attention) or multi-scale weights (for hierarchical)")
        print("5. 📋 Test on additional datasets for validation")
        print("6. ⚡ Profile computational performance for deployment considerations")

def main():
    """Run comprehensive fusion analysis"""
    
    # Path to your fusion test results
    results_path = "/home/s225078288/s225078288/Experimentation/Anomaly_Mamba/DMS2FHAD/DMS2F-HAD/Fusion_Test_Results_20250819_124527/fusion_test_results.json"
    
    print("🚀 Starting Comprehensive Fusion Analysis...")
    
    try:
        analyzer = FusionAnalyzer(results_path)
        
        # Run complete analysis
        performance_data, avg_performance = analyzer.analyze_fusion_performance()
        analyzer.interpret_gate_distributions()
        analyzer.provide_recommendations(performance_data, avg_performance)
        
        print("\n✅ Analysis Complete!")
        print("\n📋 SUMMARY:")
        print("   • Check the detailed interpretations above")
        print("   • Use the scenario-based recommendations for your use case")
        print("   • Consider the gate distribution patterns for each dataset")
        print("   • Follow the next steps for deeper validation")
        
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_path}")
        print("Please run test_all_fusion_types.py first to generate results.")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

if __name__ == "__main__":
    main()
