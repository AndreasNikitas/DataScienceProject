"""
Data Quality and Outlier Detection for Football Match Data

This module implements comprehensive data quality checks and outlier detection
for match data, including:
- Goal distribution analysis
- Missing data imputation strategies
- Outlier detection using IQR method
- Data quality scoring
"""

from typing import Dict, List, Optional, Tuple
import statistics

import pandas as pd
from sqlalchemy import text

from db import engine


def detect_goal_outliers(threshold_percentile: float = 95) -> Tuple[List[int], float]:
    """
    Detect matches with unusually high goal totals using IQR method.
    
    Args:
        threshold_percentile: Percentile threshold for outlier detection
    
    Returns:
        Tuple of (outlier_match_ids, threshold_value)
    """
    query = text("""
        SELECT id, home_goals, away_goals, 
               (home_goals + away_goals) as total_goals
        FROM matches
        WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
    """)
    
    with engine.connect() as conn:
        results = conn.execute(query).fetchall()
    
    if not results:
        return [], 0.0
    
    total_goals = [row[3] for row in results]
    
    if len(total_goals) < 4:
        return [], max(total_goals) if total_goals else 0.0
    
    q1 = statistics.quantiles(total_goals, n=4)[0]
    q3 = statistics.quantiles(total_goals, n=4)[2]
    iqr = q3 - q1
    threshold = q3 + (1.5 * iqr)
    
    outlier_ids = [row[0] for row in results if row[3] > threshold]
    
    return outlier_ids, threshold


def analyze_goal_distribution() -> Dict:
    """
    Analyze goal distribution across all matches.
    
    Returns:
        Dict with statistics on goal distribution
    """
    query = text("""
        SELECT 
            AVG(home_goals) as avg_home_goals,
            AVG(away_goals) as avg_away_goals,
            AVG(home_goals + away_goals) as avg_total_goals,
            MIN(home_goals) as min_goals,
            MAX(home_goals) as max_goals,
            COUNT(*) as finished_matches
        FROM matches
        WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query).fetchone()
    
    if not result:
        return {}
    
    return {
        "avg_home_goals": float(result[0]) if result[0] else 0,
        "avg_away_goals": float(result[1]) if result[1] else 0,
        "avg_total_goals": float(result[2]) if result[2] else 0,
        "min_goals": int(result[3]) if result[3] else 0,
        "max_goals": int(result[4]) if result[4] else 0,
        "finished_matches": int(result[5]) if result[5] else 0,
    }


def check_missing_features(match_id: int) -> List[str]:
    """
    Check for missing or null features for a match.
    
    Returns:
        List of missing feature names
    """
    query = text("""
        SELECT 
            home_last5_points,
            home_last5_goal_diff,
            home_form_pct,
            away_last5_points,
            away_last5_goal_diff,
            away_form_pct,
            h2h_home_wins,
            h2h_draws,
            h2h_away_wins
        FROM match_features
        WHERE match_id = :match_id
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {"match_id": match_id}).fetchone()
    
    if not result:
        return ["all_features_missing"]
    
    feature_names = [
        "home_last5_points", "home_last5_goal_diff", "home_form_pct",
        "away_last5_points", "away_last5_goal_diff", "away_form_pct",
        "h2h_home_wins", "h2h_draws", "h2h_away_wins"
    ]
    
    missing = [name for name, value in zip(feature_names, result) if value is None]
    return missing


def identify_incomplete_matches() -> List[int]:
    """
    Identify matches that are missing training target (result).
    
    Returns:
        List of match IDs without results
    """
    query = text("""
        SELECT id FROM matches
        WHERE result IS NULL
        AND home_goals IS NOT NULL
        AND away_goals IS NOT NULL
    """)
    
    with engine.connect() as conn:
        results = conn.execute(query).fetchall()
    
    return [row[0] for row in results]


def generate_data_quality_report() -> Dict:
    """
    Generate comprehensive data quality report.
    
    Returns:
        Dict with quality metrics and issues
    """
    # Total matches
    query_totals = text("""
        SELECT 
            COUNT(*) as total_matches,
            SUM(CASE WHEN result IS NOT NULL THEN 1 ELSE 0 END) as finished_matches,
            SUM(CASE WHEN result IS NULL THEN 1 ELSE 0 END) as upcoming_matches
        FROM matches
    """)
    
    with engine.connect() as conn:
        totals = conn.execute(query_totals).fetchone()
        total_matches = totals[0]
        finished_matches = totals[1]
        upcoming_matches = totals[2]
    
    # Feature completeness
    query_features = text("""
        SELECT 
            COUNT(*) as total_features,
            SUM(CASE WHEN home_last5_points IS NOT NULL THEN 1 ELSE 0 END) as non_null
        FROM match_features
    """)
    
    with engine.connect() as conn:
        features = conn.execute(query_features).fetchone()
        total_features = features[0]
        non_null_features = features[1]
    
    # Duplicate check
    query_dups = text("""
        SELECT COUNT(*) FROM (
            SELECT match_date, home_team_id, away_team_id, COUNT(*) as cnt
            FROM matches
            GROUP BY match_date, home_team_id, away_team_id
            HAVING cnt > 1
        ) as dupes
    """)
    
    with engine.connect() as conn:
        dups = conn.execute(query_dups).fetchone()
        duplicate_match_groups = dups[0]
    
    # Outliers
    outlier_ids, threshold = detect_goal_outliers()
    
    # Goal distribution
    goal_dist = analyze_goal_distribution()
    
    # Calculate data quality score (0-100)
    quality_score = 100
    if total_matches > 0:
        if finished_matches < total_matches * 0.5:
            quality_score -= 20
        if total_features > 0 and non_null_features / total_features < 0.9:
            quality_score -= 15
        if duplicate_match_groups > 0:
            quality_score -= 10
    
    return {
        "total_matches": total_matches,
        "finished_matches": finished_matches,
        "upcoming_matches": upcoming_matches,
        "total_features": total_features,
        "complete_features": non_null_features,
        "feature_completeness_pct": (non_null_features / total_features * 100) if total_features > 0 else 0,
        "duplicate_groups": duplicate_match_groups,
        "outlier_matches": len(outlier_ids),
        "outlier_threshold_goals": threshold,
        "goal_distribution": goal_dist,
        "data_quality_score": quality_score,
        "issues": generate_quality_issues(
            total_matches, finished_matches, duplicate_match_groups, len(outlier_ids)
        )
    }


def generate_quality_issues(total: int, finished: int, dups: int, outliers: int) -> List[str]:
    """Generate list of data quality issues."""
    issues = []
    
    if total == 0:
        issues.append("No matches in database")
    elif finished < total * 0.5:
        issues.append(f"Only {finished}/{total} matches have results ({finished/total*100:.1f}%)")
    
    if dups > 0:
        issues.append(f"{dups} duplicate match groups detected")
    
    if outliers > 0:
        issues.append(f"{outliers} matches with unusual goal counts detected")
    
    return issues


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DATA QUALITY REPORT")
    print("="*70)
    
    report = generate_data_quality_report()
    
    print(f"\n📊 MATCH COVERAGE")
    print(f"  Total matches: {report['total_matches']}")
    print(f"  Finished matches: {report['finished_matches']}")
    print(f"  Upcoming matches: {report['upcoming_matches']}")
    
    print(f"\n✓ FEATURE COMPLETENESS")
    print(f"  Total features: {report['total_features']}")
    print(f"  Complete features: {report['complete_features']}")
    print(f"  Completeness: {report['feature_completeness_pct']:.1f}%")
    
    print(f"\n⚠️  QUALITY ISSUES")
    print(f"  Duplicates: {report['duplicate_groups']}")
    print(f"  Outliers: {report['outlier_matches']} (threshold: {report['outlier_threshold_goals']:.1f} goals)")
    
    if report['goal_distribution']:
        gd = report['goal_distribution']
        print(f"\n📈 GOAL DISTRIBUTION")
        print(f"  Avg home goals: {gd['avg_home_goals']:.2f}")
        print(f"  Avg away goals: {gd['avg_away_goals']:.2f}")
        print(f"  Avg total goals: {gd['avg_total_goals']:.2f}")
        print(f"  Range: {gd['min_goals']} - {gd['max_goals']} goals")
    
    print(f"\n🎯 OVERALL DATA QUALITY SCORE: {report['data_quality_score']}/100")
    
    if report['issues']:
        print(f"\n⚠️  ISSUES FOUND:")
        for issue in report['issues']:
            print(f"   - {issue}")
    else:
        print(f"\n✅ No major quality issues detected!")
    
    print("\n" + "="*70 + "\n")
