from nba_api.stats.endpoints import commonteamroster, playergamelog, teamgamelog
from nba_api.stats.static import teams
import pandas as pd
import time
#Uses the NBA API to get the team roster for a given team and collect their statistics. If you wish to use advanced player metrics like TS%, PER, LEBRON etc. consider adding a timer since the nba api heavily rate limits users after a certain amount of requests.
# Fetch all NBA teams
nba_teams = teams.get_teams()

# Create an empty DataFrame to store all player stats and team stats
all_player_stats = pd.DataFrame()
all_team_stats = pd.DataFrame()

# Loop through each team and fetch active players
for team in nba_teams:
    team_id = team['id']
    team_name = team['full_name']
    print(f"Fetching active players and team stats for team: {team_name}")

    try:
        # Fetch the team roster for the 2024-2025 season
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season='2024-25')
        roster_data = roster.get_data_frames()[0]

        # Fetch team game logs for the 2024-2025 season
        team_gamelog = teamgamelog.TeamGameLog(team_id=team_id, season='2024-25')
        team_stats = team_gamelog.get_data_frames()[0]

        # Add team name to the team stats
        team_stats['TEAM_NAME'] = team_name

        # Append team stats to the main DataFrame
        all_team_stats = pd.concat([all_team_stats, team_stats], ignore_index=True)

        # Fetch game logs for each active player
        for _, player in roster_data.iterrows():
            player_id = player['PLAYER_ID']
            player_name = player['PLAYER']

            try:
                # Fetch game logs for the 2024-2025 season
                gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2024-25')
                player_stats = gamelog.get_data_frames()[0]

                # Add player name and team to the stats
                player_stats['PLAYER_NAME'] = player_name
                player_stats['TEAM_NAME'] = team_name

                # Append to the main DataFrame
                all_player_stats = pd.concat([all_player_stats, player_stats], ignore_index=True)
                print(f"Fetched data for {player_name} ({team_name})")
                time.sleep(0.1)
            except Exception as e:
                print(f"Failed to fetch data for {player_name}: {e}")
    except Exception as e:
        print(f"Failed to fetch roster or team stats for team {team_name}: {e}")

# Save the data to CSV files
all_player_stats.to_csv('nba_player_stats_2024_2025.csv', index=False)
all_team_stats.to_csv('nba_team_stats_2024_2025.csv', index=False)
print("Player stats saved to nba_player_stats_2024_2025.csv")
print("Team stats saved to nba_team_stats_2024_2025.csv")