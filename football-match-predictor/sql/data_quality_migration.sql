USE football_predictor;

-- Normalize team names to reduce duplicate variants.
UPDATE teams SET name = TRIM(name);
UPDATE teams SET name = REPLACE(name, '  ', ' ');
UPDATE teams SET name = REPLACE(name, '  ', ' ');
UPDATE teams SET name = REPLACE(name, '  ', ' ');

-- Merge duplicate teams by normalized name (case-insensitive).
CREATE TEMPORARY TABLE team_keep AS
SELECT MIN(id) AS keep_id, LOWER(name) AS normalized_name
FROM teams
GROUP BY LOWER(name);

CREATE TEMPORARY TABLE team_map AS
SELECT t.id AS old_id, k.keep_id
FROM teams t
JOIN team_keep k ON LOWER(t.name) = k.normalized_name;

UPDATE matches m
JOIN team_map tm ON m.home_team_id = tm.old_id
SET m.home_team_id = tm.keep_id;

UPDATE matches m
JOIN team_map tm ON m.away_team_id = tm.old_id
SET m.away_team_id = tm.keep_id;

DELETE t
FROM teams t
JOIN team_map tm ON t.id = tm.old_id
WHERE tm.old_id <> tm.keep_id;

DROP TEMPORARY TABLE team_map;
DROP TEMPORARY TABLE team_keep;

-- Remove duplicate fixtures (keep the lowest id).
DELETE m1
FROM matches m1
JOIN matches m2
  ON m1.match_date = m2.match_date
 AND m1.home_team_id = m2.home_team_id
 AND m1.away_team_id = m2.away_team_id
 AND m1.id > m2.id;

-- Remove clearly invalid rows before adding constraints.
DELETE FROM matches WHERE match_date IS NULL;
DELETE FROM matches WHERE home_team_id = away_team_id;
DELETE FROM matches WHERE home_goals < 0 OR away_goals < 0;
DELETE FROM matches WHERE home_goals > 20 OR away_goals > 20;

-- Standardize result values from scores for finished matches.
UPDATE matches
SET result = CASE
    WHEN home_goals > away_goals THEN 'H'
    WHEN away_goals > home_goals THEN 'A'
    WHEN home_goals = away_goals THEN 'D'
    ELSE NULL
END
WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL;

-- Upcoming/incomplete matches must not have a result.
UPDATE matches
SET result = NULL
WHERE home_goals IS NULL OR away_goals IS NULL;

-- Enforce stricter column types.
ALTER TABLE teams
    MODIFY name VARCHAR(100) NOT NULL;

ALTER TABLE matches
    MODIFY match_date DATETIME NOT NULL,
    MODIFY result CHAR(1) NULL;

-- Add keys/constraints (run once).
ALTER TABLE teams
    ADD UNIQUE KEY unique_team_name (name);

ALTER TABLE matches
    ADD UNIQUE KEY unique_match_fixture (match_date, home_team_id, away_team_id),
    ADD CONSTRAINT chk_different_teams CHECK (home_team_id <> away_team_id),
    ADD CONSTRAINT chk_home_goals_range CHECK (home_goals IS NULL OR (home_goals BETWEEN 0 AND 20)),
    ADD CONSTRAINT chk_away_goals_range CHECK (away_goals IS NULL OR (away_goals BETWEEN 0 AND 20)),
    ADD CONSTRAINT chk_result_value CHECK (result IS NULL OR result IN ('H', 'D', 'A')),
    ADD CONSTRAINT chk_match_completion CHECK (
        (home_goals IS NULL AND away_goals IS NULL AND result IS NULL)
        OR
        (home_goals IS NOT NULL AND away_goals IS NOT NULL AND result IS NOT NULL)
    ),
    ADD CONSTRAINT chk_result_consistency CHECK (
        result IS NULL
        OR (result = 'H' AND home_goals > away_goals)
        OR (result = 'A' AND away_goals > home_goals)
        OR (result = 'D' AND home_goals = away_goals)
    );
