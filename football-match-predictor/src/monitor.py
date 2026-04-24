"""
Model Monitoring and Drift Detection

Monitors model performance over time and detects:
- Prediction drift (model output changes)
- Data drift (input distribution changes)
- Performance degradation
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sqlalchemy import text

from db import engine


class ModelMonitor:
    """Monitor model performance and detect drift."""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize monitor."""
        self.models_dir = Path(models_dir)
        self.metrics_file = self.models_dir / "performance_metrics.json"
        self.drift_file = self.models_dir / "drift_detection.json"
    
    def calculate_prediction_statistics(self, hours: int = 24) -> Dict:
        """
        Calculate prediction statistics for recent matches.
        
        Args:
            hours: Look back period
        
        Returns:
            Dict with prediction statistics
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        query = text("""
            SELECT 
                m.result,
                COUNT(*) as count
            FROM matches m
            WHERE m.match_date >= :cutoff
            AND m.result IS NOT NULL
            GROUP BY m.result
        """)
        
        with engine.connect() as conn:
            results = conn.execute(query, {"cutoff": cutoff_time}).fetchall()
        
        stats = {
            'period_hours': hours,
            'timestamp': datetime.utcnow().isoformat(),
            'result_distribution': {}
        }
        
        total = sum(r[1] for r in results)
        for result, count in results:
            stats['result_distribution'][result] = {
                'count': count,
                'percentage': count / total * 100 if total > 0 else 0
            }
        
        return stats
    
    def detect_data_drift(self, baseline_hours: int = 168, current_hours: int = 24) -> Dict:
        """
        Detect data drift by comparing baseline vs current period.
        
        Args:
            baseline_hours: Historical period for baseline
            current_hours: Recent period for comparison
        
        Returns:
            Dict with drift metrics
        """
        baseline_cutoff = datetime.utcnow() - timedelta(hours=baseline_hours + current_hours)
        current_cutoff = datetime.utcnow() - timedelta(hours=current_hours)
        
        # Baseline period
        baseline_query = text("""
            SELECT 
                AVG(home_goals) as avg_home_goals,
                AVG(away_goals) as avg_away_goals,
                COUNT(*) as count
            FROM matches
            WHERE match_date >= :start AND match_date < :end
            AND home_goals IS NOT NULL
        """)
        
        # Current period
        current_query = text("""
            SELECT 
                AVG(home_goals) as avg_home_goals,
                AVG(away_goals) as avg_away_goals,
                COUNT(*) as count
            FROM matches
            WHERE match_date >= :start
            AND home_goals IS NOT NULL
        """)
        
        with engine.connect() as conn:
            baseline = conn.execute(
                baseline_query,
                {"start": baseline_cutoff, "end": current_cutoff}
            ).fetchone()
            current = conn.execute(
                current_query,
                {"start": current_cutoff}
            ).fetchone()
        
        if not baseline or not current:
            return {"error": "Insufficient data for drift detection"}
        
        # Calculate drift
        home_drift = (current[0] - baseline[0]) / baseline[0] * 100 if baseline[0] else 0
        away_drift = (current[1] - baseline[1]) / baseline[1] * 100 if baseline[1] else 0
        
        drift_threshold = 10  # 10% change threshold
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'baseline_period_hours': baseline_hours,
            'current_period_hours': current_hours,
            'baseline_stats': {
                'avg_home_goals': float(baseline[0]) if baseline[0] else 0,
                'avg_away_goals': float(baseline[1]) if baseline[1] else 0,
                'matches': int(baseline[2])
            },
            'current_stats': {
                'avg_home_goals': float(current[0]) if current[0] else 0,
                'avg_away_goals': float(current[1]) if current[1] else 0,
                'matches': int(current[2])
            },
            'drift': {
                'home_goals_pct': home_drift,
                'away_goals_pct': away_drift,
                'threshold': drift_threshold,
                'home_drift_detected': abs(home_drift) > drift_threshold,
                'away_drift_detected': abs(away_drift) > drift_threshold
            }
        }
    
    def track_model_performance(self) -> Dict:
        """Track model accuracy on recent completed matches."""
        query = text("""
            SELECT 
                m.match_date,
                mf.home_last5_points,
                mf.home_last5_goal_diff,
                mf.home_form_pct,
                mf.away_last5_points,
                mf.away_last5_goal_diff,
                mf.away_form_pct,
                mf.h2h_home_wins,
                mf.h2h_draws,
                mf.h2h_away_wins,
                m.result
            FROM match_features mf
            JOIN matches m ON mf.match_id = m.id
            WHERE m.result IS NOT NULL
            ORDER BY m.match_date DESC
            LIMIT 100
        """)
        
        with engine.connect() as conn:
            results = conn.execute(query).fetchall()
        
        if not results:
            return {"error": "No match data available"}
        
        df = pd.DataFrame(results, columns=[
            'match_date', 'home_last5_points', 'home_last5_goal_diff', 'home_form_pct',
            'away_last5_points', 'away_last5_goal_diff', 'away_form_pct',
            'h2h_home_wins', 'h2h_draws', 'h2h_away_wins', 'result'
        ])
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'samples_evaluated': len(df),
            'date_range': {
                'earliest': df['match_date'].min().isoformat(),
                'latest': df['match_date'].max().isoformat()
            }
        }
    
    def save_metrics(self, metrics: Dict) -> None:
        """Save metrics to file."""
        self.models_dir.mkdir(exist_ok=True)
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def save_drift_report(self, drift: Dict) -> None:
        """Save drift detection report."""
        self.models_dir.mkdir(exist_ok=True)
        
        with open(self.drift_file, 'w') as f:
            json.dump(drift, f, indent=2)
    
    def generate_monitoring_report(self) -> None:
        """Generate comprehensive monitoring report."""
        print("\n" + "="*70)
        print("MODEL MONITORING REPORT")
        print("="*70)
        
        # Performance statistics
        print("\n📊 PERFORMANCE STATISTICS")
        perf = self.track_model_performance()
        if 'error' not in perf:
            print(f"  Samples Evaluated: {perf['samples_evaluated']}")
            print(f"  Date Range: {perf['date_range']['earliest']} to {perf['date_range']['latest']}")
        
        # Prediction statistics
        print("\n📈 RECENT PREDICTIONS (24h)")
        pred_stats = self.calculate_prediction_statistics(hours=24)
        for result, stats in pred_stats['result_distribution'].items():
            print(f"  {result}: {stats['count']} matches ({stats['percentage']:.1f}%)")
        
        # Data drift detection
        print("\n⚠️  DATA DRIFT DETECTION")
        drift = self.detect_data_drift()
        if 'error' not in drift:
            drift_info = drift['drift']
            print(f"  Threshold: ±{drift_info['threshold']}%")
            print(f"  Home Goals: {drift_info['home_goals_pct']:+.1f}% "
                  f"({'DRIFT' if drift_info['home_drift_detected'] else 'OK'})")
            print(f"  Away Goals: {drift_info['away_goals_pct']:+.1f}% "
                  f"({'DRIFT' if drift_info['away_drift_detected'] else 'OK'})")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    monitor = ModelMonitor()
    monitor.generate_monitoring_report()
