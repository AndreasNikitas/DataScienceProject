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
    
    FOREIGN KEY (match_id) REFERENCES matches(id)
);
