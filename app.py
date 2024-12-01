import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from statsbombpy import sb
from mplsoccer import Pitch
import streamlit as st

# Helper Functions
def preprocess_event_data(event_df):
    """Preprocess event data: expand location columns and calculate parts of the pitch."""
    location_columns = ['shot_end_location', 'goalkeeper_end_location', 'carry_end_location', 'location', 'pass_end_location']
    for col in location_columns:
        event_df[f'{col}_x'] = event_df[col].apply(lambda loc: loc[0] if isinstance(loc, (list, tuple)) and len(loc) > 0 else None)
        event_df[f'{col}_y'] = event_df[col].apply(lambda loc: loc[1] if isinstance(loc, (list, tuple)) and len(loc) > 1 else None)
        event_df[f'{col}_z'] = event_df[col].apply(lambda loc: loc[2] if isinstance(loc, (list, tuple)) and len(loc) > 2 else None)
    event_df['Part_of_pitch'] = np.where(event_df['location_x'] >= 80, 'Attacking-3rd',
                                         np.where(event_df['location_x'] <= 40, 'Defensive-3rd', 'Mid-field'))
    return event_df

def extract_starting_lineup(event_df):
    """Extract the starting XI and formations for both teams."""
    lineup_df = event_df[event_df.type == 'Starting XI'].reset_index(drop=True)
    team0, team1 = lineup_df.iloc[0]['team'], lineup_df.iloc[1]['team']
    formation0, formation1 = lineup_df.iloc[0]['tactics']['formation'], lineup_df.iloc[1]['tactics']['formation']
    return team0, team1, formation0, formation1

def expand_lineup_data(lineup_df, team_name):
    """Expand and format lineup data for a given team."""
    team_lineup = sb.lineups(match_id=selected_match_id)[team_name]
    positions = team_lineup['positions'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else {}).apply(pd.Series)
    cards = team_lineup['cards'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else {}).apply(pd.Series)
    expanded_lineup = pd.concat([team_lineup, positions, cards], axis=1).drop(columns=['positions', 'cards'])
    expanded_lineup.sort_values(by=['start_reason', 'end_reason'], inplace=True)
    columns = ['player_name', 'jersey_number', 'position', 'from', 'to', 'start_reason', 'end_reason']
    if 'card_type' in expanded_lineup.columns:
        columns += ['card_type', 'time', 'reason']
    return expanded_lineup[columns]

def calculate_possession(event_df, team0, team1):
    """Calculate possession percentages for both teams."""
    total_possessions = event_df.possession_team.count()
    poss0 = event_df[event_df.possession_team == team0]['possession_team'].count() / total_possessions
    poss1 = event_df[event_df.possession_team == team1]['possession_team'].count() / total_possessions
    return poss0, poss1

def plot_possession_chart(poss0, poss1, team0, team1):
    """Plot a horizontal bar chart for possession percentages."""
    fig, ax = plt.subplots(figsize=(18, 1))
    ax.barh(y=0, width=poss0, color='#DEEFF5', label=f'{team0} - {poss0 * 100:.1f}%')
    ax.barh(y=0, width=poss1, left=poss0, color='#90EE90', label=f'{team1} - {poss1 * 100:.1f}%')
    ax.axis('off')
    ax.text(poss0 / 2, 0, f"{team0} - {poss0 * 100:.1f}%", fontsize=14, ha='center', va='center', fontweight='bold', color='black', fontname="Monospace")
    ax.text(poss0 + poss1 / 2, 0, f"{team1} - {poss1 * 100:.1f}%", fontsize=14, ha='center', va='center', fontweight='bold', color='black', fontname="Monospace")
    plt.suptitle("Possession", fontsize=14, fontweight='bold', fontname="Monospace", y=1.1)
    return fig

def plot_formation(pitch, avg_locations, formation, team_name, team_color, fig_size=(10, 6)):
    """Plot average player positions as formation."""
    fig, ax = plt.subplots(figsize=fig_size)
    pitch.draw(ax=ax)
    for i, row in avg_locations.iterrows():
        pitch.scatter(row['location_x'], row['location_y'], color=team_color, edgecolors='black', s=600, ax=ax)
        pitch.text(row['location_x'], row['location_y'], s=row['jersey_number'], color='black', weight='bold', ha='center',
                   va='center', fontsize=15, fontname="Monospace", zorder=2, ax=ax)
    ax.set_title(f'{team_name} - {formation}', fontsize=14, fontweight='bold', fontname="Monospace", y=0.97)
    return fig

# Streamlit Setup
st.title("Match Report by Precious")

# Load competition data and select a competition
comp_df = sb.competitions()
comp_df['Tournament'] = comp_df.competition_name + ' ' + comp_df.season_name
comp_list = comp_df.Tournament.unique().tolist()
comp = st.selectbox("Select Competition:", comp_list)

if comp:
    c_id = comp_df.loc[comp_df['Tournament'] == comp, 'competition_id'].values[0]
    s_id = comp_df.loc[comp_df['Tournament'] == comp, 'season_id'].values[0]
    matches_df = sb.matches(competition_id=c_id, season_id=s_id)
    matches_df['Match'] = matches_df['home_team'] + '(' + matches_df['home_score'].astype(str) + ')' + ' vs ' + matches_df['away_team'] + '(' + matches_df['away_score'].astype(str) + ')'
    match = st.selectbox("Select Match:", matches_df.Match.unique().tolist())

    if match:
        selected_match_id = matches_df.loc[matches_df['Match'] == match, 'match_id'].values[0]
        event_df = sb.events(match_id=int(selected_match_id))
        event_df = preprocess_event_data(event_df)

        # Extract and process data
        team0, team1, formation0, formation1 = extract_starting_lineup(event_df)
        lineup0 = expand_lineup_data(event_df, team0)
        lineup1 = expand_lineup_data(event_df, team1)
        poss0, poss1 = calculate_possession(event_df, team0, team1)

        # Display possession chart
        fig = plot_possession_chart(poss0, poss1, team0, team1)
        st.pyplot(fig)

        # Display lineups and formations
        st.subheader(f"{team0} XI")
        st.dataframe(lineup0)
        st.subheader(f"{team1} XI")
        st.dataframe(lineup1)

        st.markdown("<h2 style='text-align: center;'>Average Formation</h2>", unsafe_allow_html=True)
        avg_locations0 = event_df[event_df.team == team0].groupby('player')[['location_x', 'location_y']].mean().reset_index()
        avg_locations1 = event_df[event_df.team == team1].groupby('player')[['location_x', 'location_y']].mean().reset_index()

        pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
        fig0 = plot_formation(pitch, avg_locations0, formation0, team0, '#DEEFF5')
        fig1 = plot_formation(pitch, avg_locations1, formation1, team1, '#90EE90')

        st.pyplot(fig0)
        st.pyplot(fig1)

        # Add more sections for pass analysis, shots, etc., using similar modular approaches
