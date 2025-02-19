import pandas as pd
import requests
from bs4 import BeautifulSoup
from nba_api.stats.endpoints import commonteamroster, playergamelog
from nba_api.stats.static import teams
import time
from io import StringIO

def scrape_basketball_ref_stats():
    """Scrape both offensive and defensive team stats from Basketball Reference"""
    base_url = "https://www.basketball-reference.com/leagues/NBA_2025.html"
    
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Scrape offensive stats (Team Per Game)
        off_table = soup.find('table', {'id': 'per_game-team'})
        off_df = pd.read_html(StringIO(str(off_table)))[0]
        off_df = off_df[off_df['Team'] != 'Team'].dropna(axis=1, how='all')
        off_df.columns = [f"OFF_{col}" if col not in ['Rk', 'Team'] else col for col in off_df.columns]

        # Scrape defensive stats (Opponent Per Game)
        def_table = soup.find('table', {'id': 'per_game-opponent'})
        def_df = pd.read_html(StringIO(str(def_table)))[0]
        def_df = def_df[def_df['Team'] != 'Team'].dropna(axis=1, how='all')
        def_df.columns = [f"DEF_{col}" if col not in ['Rk', 'Team'] else col for col in def_df.columns]

        # Merge and clean data
        merged_df = pd.merge(
            off_df,
            def_df,
            on='Team',
            suffixes=('_off', '_def')
        )

        # Clean team names and set index
        merged_df['Team'] = merged_df['Team'].str.replace('*', '', regex=False)
        return merged_df.drop(columns=['Rk_off', 'Rk_def']).reset_index(drop=True)

    except Exception as e:
        print(f"Error scraping Basketball Reference: {e}")
        return pd.DataFrame()

def fetch_nba_player_stats():
    """Fetch player game logs from NBA API"""
    all_players = pd.DataFrame()
    nba_teams = teams.get_teams()
    
    for team in nba_teams:
        team_id = team['id']
        team_name = team['full_name']
        print(f"Processing {team_name}...")
        
        try:
            # Get team roster
            roster = commonteamroster.CommonTeamRoster(team_id=team_id, season='2024-25')
            players = roster.get_data_frames()[0]
            
            # Get player game logs
            for _, player in players.iterrows():
                player_id = player['PLAYER_ID']
                player_name = player['PLAYER']
                
                try:
                    gamelog = playergamelog.PlayerGameLog(
                        player_id=player_id, 
                        season='2024-25',
                        season_type_all_star='Regular Season'
                    )
                    player_stats = gamelog.get_data_frames()[0]
                    
                    # Add player name and team name to the stats
                    player_stats['PLAYER_NAME'] = player_name
                    player_stats['TEAM'] = team_name
                    
                    # Concatenate with all players
                    all_players = pd.concat([all_players, player_stats], ignore_index=True)
                    time.sleep(0.6)  # Rate limit protection
                except Exception as e:
                    print(f"Error fetching {player_name}: {str(e)[:100]}...")
                    
        except Exception as e:
            print(f"Error processing {team_name}: {str(e)[:100]}...")
    
    # Standardize column names
    all_players.rename(columns={'GAME_DATE': 'DATE', 'MATCHUP': 'OPPONENT'}, inplace=True)
    return all_players

def main():
    # Scrape team stats from Basketball Reference
    print("Scraping team stats from Basketball Reference...")
    team_stats = scrape_basketball_ref_stats()
    
    # Fetch player stats from NBA API
    print("\nFetching player stats from NBA API...")
    player_stats = fetch_nba_player_stats()
    
    # Save data
    team_stats.to_csv('nba_team_stats_2024_2025.csv', index=False)
    player_stats.to_csv('nba_player_stats_2024_2025.csv', index=False)
    
    print(f"\nSuccessfully saved data!")
    print(f"Team stats shape: {team_stats.shape}")
    print(f"Player stats shape: {player_stats.shape}")

if __name__ == "__main__":
    main()