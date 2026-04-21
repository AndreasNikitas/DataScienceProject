CREATE DATABASE IF NOT EXISTS football_predictor;
USE football_predictor;

CREATE TABLE IF NOT EXISTS teams (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    UNIQUE KEY unique_team_name (name)
);

CREATE TABLE IF NOT EXISTS matches (
    id INT PRIMARY KEY AUTO_INCREMENT,
    match_date DATETIME NOT NULL,
    home_team_id INT NOT NULL,
    away_team_id INT NOT NULL,
    home_goals INT NULL,
    away_goals INT NULL,
    result CHAR(1) NULL,
    CONSTRAINT chk_different_teams CHECK (home_team_id <> away_team_id),
    CONSTRAINT chk_home_goals_range CHECK (home_goals IS NULL OR (home_goals BETWEEN 0 AND 20)),
    CONSTRAINT chk_away_goals_range CHECK (away_goals IS NULL OR (away_goals BETWEEN 0 AND 20)),
    CONSTRAINT chk_result_value CHECK (result IS NULL OR result IN ('H', 'D', 'A')),
    CONSTRAINT chk_match_completion CHECK (
        (home_goals IS NULL AND away_goals IS NULL AND result IS NULL)
        OR
        (home_goals IS NOT NULL AND away_goals IS NOT NULL AND result IS NOT NULL)
    ),
    CONSTRAINT chk_result_consistency CHECK (
        result IS NULL
        OR (result = 'H' AND home_goals > away_goals)
        OR (result = 'A' AND away_goals > home_goals)
        OR (result = 'D' AND home_goals = away_goals)
    ),
    UNIQUE KEY unique_match_fixture (match_date, home_team_id, away_team_id),
    FOREIGN KEY (home_team_id) REFERENCES teams(id),
    FOREIGN KEY (away_team_id) REFERENCES teams(id)
);

CREATE TABLE IF NOT EXISTS match_features (
    match_id INT PRIMARY KEY,
    home_last5_points INT DEFAULT 0,
    home_last5_goal_diff INT DEFAULT 0,
    home_form_pct DECIMAL(5,2) DEFAULT 0,
    away_last5_points INT DEFAULT 0,
    away_last5_goal_diff INT DEFAULT 0,
    away_form_pct DECIMAL(5,2) DEFAULT 0,
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

CREATE TABLE IF NOT EXISTS prediction_runs (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    league_slug VARCHAR(32) NOT NULL DEFAULT 'eng.1',
    total_matches INT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS match_predictions (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    run_id BIGINT NOT NULL,
    match_id INT NOT NULL,
    predicted_result CHAR(1) NOT NULL,
    predicted_home_goals INT NOT NULL,
    predicted_away_goals INT NOT NULL,
    rf_prediction CHAR(1) NULL,
    rf_confidence DECIMAL(6,5) NULL,
    lr_prediction CHAR(1) NULL,
    lr_confidence DECIMAL(6,5) NULL,
    consensus_prediction CHAR(1) NULL,
    actual_home_goals INT NULL,
    actual_away_goals INT NULL,
    actual_result CHAR(1) NULL,
    outcome_correct TINYINT(1) NULL,
    score_exact TINYINT(1) NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'pending',
    resolved_at DATETIME NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    CONSTRAINT chk_prediction_result_value CHECK (predicted_result IN ('H', 'D', 'A')),
    CONSTRAINT chk_actual_result_value CHECK (actual_result IS NULL OR actual_result IN ('H', 'D', 'A')),
    CONSTRAINT chk_prediction_status CHECK (status IN ('pending', 'resolved')),
    UNIQUE KEY unique_run_match_prediction (run_id, match_id),
    KEY idx_predictions_match (match_id),
    KEY idx_predictions_status (status),
    FOREIGN KEY (run_id) REFERENCES prediction_runs(id),
    FOREIGN KEY (match_id) REFERENCES matches(id)
);
