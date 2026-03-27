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
