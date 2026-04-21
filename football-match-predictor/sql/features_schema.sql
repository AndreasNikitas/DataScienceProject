-- Add team features table for ML predictions
-- This stores calculated statistics for each team

USE football_predictor;

CREATE TABLE IF NOT EXISTS team_form (
    id INT PRIMARY KEY AUTO_INCREMENT,
    team_id INT NOT NULL,
    calculated_date DATE NOT NULL,
    
    -- Form metrics (last 5 matches)
    last5_wins INT DEFAULT 0,
    last5_draws INT DEFAULT 0,
    last5_losses INT DEFAULT 0,
    last5_points INT DEFAULT 0,
    
    -- Goal metrics (last 5 matches)
    last5_goals_scored INT DEFAULT 0,
    last5_goals_conceded INT DEFAULT 0,
    last5_goal_diff INT DEFAULT 0,
    
    -- Overall season stats
    total_matches INT DEFAULT 0,
    total_wins INT DEFAULT 0,
    total_draws INT DEFAULT 0,
    total_losses INT DEFAULT 0,
    
    FOREIGN KEY (team_id) REFERENCES teams(id),
    UNIQUE KEY unique_team_date (team_id, calculated_date)
);

-- Add match features table
-- This stores features for each match (before it's played)
CREATE TABLE IF NOT EXISTS match_features (
    match_id INT PRIMARY KEY,
    
    -- Home team features
    home_last5_points INT DEFAULT 0,
    home_last5_goal_diff INT DEFAULT 0,
    home_form_pct DECIMAL(5,2) DEFAULT 0,
    
    -- Away team features  
    away_last5_points INT DEFAULT 0,
    away_last5_goal_diff INT DEFAULT 0,
    away_form_pct DECIMAL(5,2) DEFAULT 0,
    
    -- Head-to-head
    h2h_home_wins INT DEFAULT 0,
    h2h_draws INT DEFAULT 0,
    h2h_away_wins INT DEFAULT 0,
    home_days_since_last_match DECIMAL(6,2) DEFAULT 14.0,
    away_days_since_last_match DECIMAL(6,2) DEFAULT 14.0,
    home_matches_last7 INT DEFAULT 0,
    away_matches_last7 INT DEFAULT 0,
    home_matches_last14 INT DEFAULT 0,
    away_matches_last14 INT DEFAULT 0,
    home_travel_penalty DECIMAL(5,3) DEFAULT 0.000,
    away_travel_penalty DECIMAL(5,3) DEFAULT 0.000,
    home_home_ppg_last10 DECIMAL(6,3) DEFAULT 0.000,
    away_away_ppg_last10 DECIMAL(6,3) DEFAULT 0.000,
    home_home_goal_diff_last10 DECIMAL(6,3) DEFAULT 0.000,
    away_away_goal_diff_last10 DECIMAL(6,3) DEFAULT 0.000,
    home_overall_ppg_last30 DECIMAL(6,3) DEFAULT 0.000,
    away_overall_ppg_last30 DECIMAL(6,3) DEFAULT 0.000,
    home_overall_goal_diff_last30 DECIMAL(6,3) DEFAULT 0.000,
    away_overall_goal_diff_last30 DECIMAL(6,3) DEFAULT 0.000,
    home_elo DECIMAL(8,3) DEFAULT 1500.0,
    away_elo DECIMAL(8,3) DEFAULT 1500.0,
    elo_diff DECIMAL(8,3) DEFAULT 0.0,
    
    FOREIGN KEY (match_id) REFERENCES matches(id)
);
