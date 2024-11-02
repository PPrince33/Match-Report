import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsbombpy import sb
from mplsoccer import Pitch

# Fetch competitions data
comp_df = sb.competitions()
comp_df['Tournament'] = comp_df.competition_name + ' ' + comp_df.season_name
comp_list = comp_df.Tournament.unique().tolist()

# Streamlit title and competition selection
st.title("Match Report by Precious")
comp = st.selectbox("Select Competition:", comp_list)

if comp:
    c_id = comp_df.loc[comp_df['Tournament'] == comp, 'competition_id'].values[0]
    s_id = comp_df.loc[comp_df['Tournament'] == comp, 'season_id'].values[0]
    matches_df = sb.matches(competition_id=c_id, season_id=s_id)
    matches_df['Match'] = matches_df['home_team'] + '(' + matches_df['home_score'].astype(str) + ')' + ' vs ' + matches_df['away_team'] + '(' + matches_df['away_score'].astype(str) + ')'
    match_list = matches_df.Match.unique().tolist()
    match = st.selectbox("Select Match:", match_list)
    
    if match:
        match_id = matches_df.loc[matches_df['Match'] == match, 'match_id'].values[0]
        event_df = sb.events(match_id=int(match_id))
        event_df[['X', 'Y']] = event_df['location'].apply(pd.Series)
        event_df[['endX', 'endY']] = event_df['pass_end_location'].apply(pd.Series)        
        event_df['Part_of_pitch'] = np.where((event_df['X'] >= 80) & (event_df['endX'] >= 80), 'Attacking-3rd',
                                              np.where((event_df['X'] <= 40) & (event_df['endX'] <= 40), 'Defensive-3rd', 'Mid-field'))
        
        pitch = Pitch(pitch_type='statsbomb', pitch_color='black', line_color='white')
        fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)
        fig.set_facecolor("black")

        st.header("Lineup")
        team_name0 = event_df.at[0, 'team']    
        team_name1 = event_df.at[1, 'team']
        lineup_data0 = sb.lineups(match_id=int(match_id))[team_name0][['player_name', 'player_id', 'jersey_number']]
        lineup_data1 = sb.lineups(match_id=int(match_id))[team_name1][['player_name', 'player_id', 'jersey_number']]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{team_name0}")
            st.dataframe(lineup_data0)
        with col2:
            st.subheader(f"{team_name1}")
            st.dataframe(lineup_data1)

        # Pass analysis
        pass_df0 = event_df[(event_df.type == 'Pass') & (event_df.team == team_name0)]
        pass_df1 = event_df[(event_df.type == 'Pass') & (event_df.team == team_name1)]

        # Completed passes for both teams
        completed_passes_team0 = pass_df0[pass_df0['pass_outcome'].isnull()]
        completed_passes_team1 = pass_df1[pass_df1['pass_outcome'].isnull()]
        
        # First Substitution
        subs0 = event_df[(event_df['type'] == 'Substitution') & (event_df['team'] == team_name0)]
        firstsub0 = subs0['minute'].min() if not subs0.empty else np.inf

        subs1 = event_df[(event_df['type'] == 'Substitution') & (event_df['team'] == team_name1)]
        firstsub1 = subs1['minute'].min() if not subs1.empty else np.inf

        # Filter successful passes before first substitution
        successful0 = completed_passes_team0[completed_passes_team0['minute'] < firstsub0]
        successful1 = completed_passes_team1[completed_passes_team1['minute'] < firstsub1]

        # Add jersey numbers
        successful0 = pd.merge(successful0, lineup_data0, on='player_id', how='left)
        successful1 = pd.merge(successful1, lineup_data1, on='player_id', how='left')

        successful0.rename(columns={'player_name': 'passer_name', 'jersey_number': 'passer_jersey_no'}, inplace=True)
        successful1.rename(columns={'player_name': 'passer_name', 'jersey_number': 'passer_jersey_no'}, inplace=True)

        # Prepare average locations
        avg_locations0 = successful0.groupby('passer_jersey_no').agg({'X': 'mean', 'Y': 'mean', 'minute': 'count'}).reset_index()
        avg_locations0.rename(columns={'minute': 'count'}, inplace=True)

        avg_locations1 = successful1.groupby('passer_jersey_no').agg({'X': 'mean', 'Y': 'mean', 'minute': 'count'}).reset_index()
        avg_locations1.rename(columns={'minute': 'count'}, inplace=True)

        # Passes between players for plotting
        pass_between0 = successful0.groupby(['passer_jersey_no', 'recipient_jersey_no']).size().reset_index(name='pass_count')
        pass_between1 = successful1.groupby(['passer_jersey_no', 'recipient_jersey_no']).size().reset_index(name='pass_count')

        # Plot passing network for Team 0
        pitch0 = Pitch(pitch_type='statsbomb', pitch_color='#FFDC02', line_color='black')
        fig0, ax0 = pitch0.draw(figsize=(8, 11), constrained_layout=True)
        fig0.set_facecolor("black")

        for _, row in pass_between0.iterrows():
            passer = row['passer_jersey_no']
            recipient = row['recipient_jersey_no']
            pass_count = row['pass_count']
            start_loc = avg_locations0[avg_locations0['passer_jersey_no'] == passer][['X', 'Y']].values[0]
            end_loc = avg_locations0[avg_locations0['passer_jersey_no'] == recipient][['X', 'Y']].values[0]
            pitch0.lines(start_loc[0], start_loc[1], end_loc[0], end_loc[1], lw=pass_count, color="#193375", zorder=0.7, ax=ax0)

        pitch0.scatter(avg_locations0['X'], avg_locations0['Y'], s=30 * avg_locations0['count'].values, color='#19AE47', edgecolors='black', linewidth=1, ax=ax0)

        for index, row in avg_locations0.iterrows():
            pitch0.annotate(row['passer_jersey_no'], xy=(row['X'], row['Y']), color='#161A30', va='center', ha='center', size=15, ax=ax0)

        ax0.set_title(f'{team_name0} Passing Network', color='white', fontsize=20)

        # Plot passing network for Team 1
        pitch1 = Pitch(pitch_type='statsbomb', pitch_color='#FFDC02', line_color='black')
        fig1, ax1 = pitch1.draw(figsize=(8, 11), constrained_layout=True)
        fig1.set_facecolor("black")

        for _, row in pass_between1.iterrows():
            passer = row['passer_jersey_no']
            recipient = row['recipient_jersey_no']
            pass_count = row['pass_count']
            start_loc = avg_locations1[avg_locations1['passer_jersey_no'] == passer][['X', 'Y']].values[0]
            end_loc = avg_locations1[avg_locations1['passer_jersey_no'] == recipient][['X', 'Y']].values[0]
            pitch1.lines(start_loc[0], start_loc[1], end_loc[0], end_loc[1], lw=pass_count, color="#193375", zorder=0.7, ax=ax1)

        pitch1.scatter(avg_locations1['X'], avg_locations1['Y'], s=30 * avg_locations1['count'].values, color='#19AE47', edgecolors='black', linewidth=1, ax=ax1)

        for index, row in avg_locations1.iterrows():
            pitch1.annotate(row['passer_jersey_no'], xy=(row['X'], row['Y']), color='#161A30', va='center', ha='center', size=15, ax=ax1)

        ax1.set_title(f'{team_name1} Passing Network', color='white', fontsize=20)

        # Display plots
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig0)
        with col2:
            st.pyplot(fig1)

        # Pass Table
        pass_table = pd.DataFrame(columns=[team_name0, team_name1])
        completed_pass_count0 = pass_df0[pass_df0['pass_outcome'].isnull()]['passer_jersey_no'].value_counts()
        completed_pass_count1 = pass_df1[pass_df1['pass_outcome'].isnull()]['passer_jersey_no'].value_counts()
        pass_table[team_name0] = completed_pass_count0
        pass_table[team_name1] = completed_pass_count1
        st.header("Completed Passes Table")
        st.dataframe(pass_table)

        # Clean up
        st.balloons()
