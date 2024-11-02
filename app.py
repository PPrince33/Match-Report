import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsbombpy import sb
from mplsoccer import Pitch

comp_df=sb.competitions()
comp_df['Tournament']=comp_df.competition_name + ' ' + comp_df.season_name
comp_list=comp_df.Tournament.unique().tolist()

st.title("Match Report by Precious")
comp = st.selectbox("Select Competition:", comp_list)

if comp:
    c_id = comp_df.loc[comp_df['Tournament'] == comp, 'competition_id'].values[0]
    s_id = comp_df.loc[comp_df['Tournament'] == comp, 'season_id'].values[0]
    matches_df=sb.matches(competition_id=c_id,season_id=s_id)
    matches_df['Match']=matches_df['home_team'] + '(' + matches_df['home_score'].astype(str) + ')' + ' vs ' + matches_df['away_team'] + '(' + matches_df['away_score'].astype(str) + ')'
    match_list=matches_df.Match.unique().tolist()
    match=st.selectbox("Select Match:", match_list)
    if match:
        match_id=matches_df.loc[matches_df['Match']==match,'match_id'].values[0]
        event_df=sb.events(match_id=int(match_id))
        event_df[['X', 'Y']] = event_df['location'].apply(pd.Series)
        event_df[['endX', 'endY']] = event_df['pass_end_location'].apply(pd.Series)        
        event_df['Part_of_pitch']=np.where((event_df['X']>=80) & (event_df['endX']>=80) ,'Attacking-3rd',np.where((event_df['X']<=40) & (event_df['endX']<=40) ,'Defensive-3rd','Mid-field'))
        
        pitch = Pitch(pitch_type='statsbomb', pitch_color='black', line_color='white')
        fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)
        fig.set_facecolor("black")
        
        
        
        
        
        
        st.header("Starting XI")      
        lineup_data0 = pd.DataFrame(event_df.at[0, 'tactics']['lineup'])
        lineup_data0['player_name'] = lineup_data0['player'].apply(lambda x: x['name'])
        lineup_data0['position_name'] = lineup_data0['position'].apply(lambda x: x['name'])
        lineup_data0 = lineup_data0[['player_name', 'position_name', 'jersey_number']]
        team_name0 = event_df.at[0, 'team']    
        lineup_data1 = pd.DataFrame(event_df.at[1, 'tactics']['lineup'])
        lineup_data1['player_name'] = lineup_data1['player'].apply(lambda x: x['name'])
        lineup_data1['position_name'] = lineup_data1['position'].apply(lambda x: x['name'])
        lineup_data1 = lineup_data1[['player_name', 'position_name', 'jersey_number']]
        team_name1 = event_df.at[1, 'team']
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{team_name0}")
            st.dataframe(lineup_data0)
            #st.markdown(
    f'<div style="text-align: center;">{lineup_data0.to_html(index=False)}</div>',
    unsafe_allow_html=True
)
        with col2:
            st.subheader(f"{team_name1}")
            st.dataframe(lineup_data1)
            #st.markdown(
    f'<div style="text-align: center;">{lineup_data1.to_html(index=False)}</div>',
    unsafe_allow_html=True
)
        
        
        #Pass
        pass_df0=event_df[(event_df.type=='Pass')&(event_df.team==team_name0)]
        pass_df1=event_df[(event_df.type=='Pass')&(event_df.team==team_name1)]
        pass_table=pd.DataFrame(columns=[team_name0,team_name1])

        # Filter passes where 'pass_outcome' is null for both DataFrames
        completed_passes_team1 = pass_df0[pass_df0['pass_outcome'].isnull()]
        completed_passes_team2 = pass_df1[pass_df1['pass_outcome'].isnull()]
        
        # Unique options for Part_of_pitch and player for both DataFrames
        part_of_pitch_options0 = ['All'] + completed_passes_team1['Part_of_pitch'].unique().tolist()
        part_of_pitch_options1 = ['All'] + completed_passes_team2['Part_of_pitch'].unique().tolist()
        player_options0 = ['All'] + completed_passes_team1['player'].unique().tolist()
        player_options1 = ['All'] + completed_passes_team2['player'].unique().tolist()
        
        # Team 1 filters (completed_passes_team1)
        st.sidebar.subheader(f"{team_name0} Filters")
        part_of_pitch_selected0 = st.sidebar.selectbox(f"Select Part of Pitch {team_name0}", options=part_of_pitch_options0)
        players_selected0 = st.sidebar.selectbox(f"Select Player(s) {team_name0}", options=player_options0)
        minute_slider0 = st.sidebar.slider(f"Select Minute Range {team_name0}", min_value=int(completed_passes_team1['minute'].min()), max_value=int(completed_passes_team1['minute'].max()), value=(int(completed_passes_team1['minute'].min()), int(completed_passes_team1['minute'].max())))
        
        # Team 2 filters (completed_passes_team2)
        st.sidebar.subheader(f"{team_name1} Filters")
        part_of_pitch_selected1 = st.sidebar.selectbox(f"Select Part of Pitch {team_name1}", options=part_of_pitch_options1)
        players_selected1 = st.sidebar.selectbox(f"Select Player(s) {team_name1}", options=player_options1)
        minute_slider1 = st.sidebar.slider(f"Select Minute Range {team_name1}", min_value=int(completed_passes_team2['minute'].min()), max_value=int(completed_passes_team2['minute'].max()), value=(int(completed_passes_team2['minute'].min()), int(completed_passes_team2['minute'].max())))
        
        # Apply filters to completed_passes_team1 based on selected Part of Pitch and Players
        if part_of_pitch_selected0 != 'All':
            completed_passes_team1 = completed_passes_team1[completed_passes_team1['Part_of_pitch'] == part_of_pitch_selected0]
        if players_selected0 != 'All':
            completed_passes_team1 = completed_passes_team1[completed_passes_team1['player'] == players_selected0]
        
        # Filter based on minute range for Team 1
        completed_passes_team1 = completed_passes_team1[(completed_passes_team1['minute'] >= minute_slider0[0]) & (completed_passes_team1['minute'] <= minute_slider0[1])]
        
        # Apply filters to completed_passes_team2 based on selected Part of Pitch and Players
        if part_of_pitch_selected1 != 'All':
            completed_passes_team2 = completed_passes_team2[completed_passes_team2['Part_of_pitch'] == part_of_pitch_selected1]
        if players_selected1 != 'All':
            completed_passes_team2 = completed_passes_team2[completed_passes_team2['player'] == players_selected1]
        
        # Filter based on minute range for Team 2
        completed_passes_team2 = completed_passes_team2[(completed_passes_team2['minute'] >= minute_slider1[0]) & (completed_passes_team2['minute'] <= minute_slider1[1])]
        
        # Initialize the pitch settings
        pitch = Pitch(pitch_type='statsbomb', pitch_color='black', line_color='white')
        
        # Streamlit layout for side-by-side pitch maps
        col1, col2 = st.columns(2)
        
        # Plot for Team 1 with filters applied
        with col1:
            fig, ax = pitch.draw(figsize=(8, 6), constrained_layout=True, tight_layout=False)
            fig.set_facecolor("black")
            # Draw lines for each pass
            for idx, row in completed_passes_team1.iterrows():
                ax.plot([row['X'], row['endX']], [row['Y'], row['endY']], color='white',linestyle='--', linewidth=1)  # Line between passes
            ax.scatter(completed_passes_team1['X'], completed_passes_team1['Y'], color='green', label="Start")
            ax.scatter(completed_passes_team1['endX'], completed_passes_team1['endY'], color='red', label="End")
            ax.legend(loc="upper left")
            st.pyplot(fig)
        
        # Plot for Team 2 with filters applied
        with col2:
            fig, ax = pitch.draw(figsize=(8, 6), constrained_layout=True, tight_layout=False)
            fig.set_facecolor("black")
            # Draw lines for each pass
            for idx, row in completed_passes_team2.iterrows():
                ax.plot([row['X'], row['endX']], [row['Y'], row['endY']], color='white', linestyle='--',linewidth=1)  # Line between passes
            ax.scatter(completed_passes_team2['X'], completed_passes_team2['Y'], color='green', label="Start")
            ax.scatter(completed_passes_team2['endX'], completed_passes_team2['endY'], color='red', label="End")
            ax.legend(loc="upper left")
            st.pyplot(fig)
        
        st.markdown("**Note:** Mapping only includes completed passes.")

        
        #Pass Table
        completed_pass_count0 = pass_df0[pass_df0['pass_outcome'].isnull()].id.count()
        completed_pass_count1 = pass_df1[pass_df1['pass_outcome'].isnull()].id.count()
        pass_table.loc['Completed Pass', team_name0] = completed_pass_count0
        pass_table.loc['Completed Pass', team_name1] = completed_pass_count1
        pass_type_counts0 = pass_df0['pass_type'].value_counts()
        for pass_type, count in pass_type_counts0.items():
            pass_table.loc[pass_type, team_name0] = count
        pass_type_counts1 = pass_df1['pass_type'].value_counts()
        for pass_type, count in pass_type_counts1.items():
            pass_table.loc[pass_type, team_name1] = count
        pass_outcome_counts0 = pass_df0['pass_outcome'].value_counts()
        pass_outcome_counts1 = pass_df1['pass_outcome'].value_counts()
        for outcome, count in pass_outcome_counts0.items():
            pass_table.loc[outcome, team_name0] = count
        for outcome, count in pass_outcome_counts1.items():
            pass_table.loc[outcome, team_name1] = count
        pass_table = pass_table.fillna(0)
        
        pitch = Pitch(pitch_type='statsbomb', pitch_color='black', line_color='white')
        fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)
        fig.set_facecolor("black")
        
        
        
        
        pass_table=pass_table.reset_index().rename(columns={'index':'Paticular'})




        
        st.subheader("Pass Analysis")
        st.dataframe(pass_table)
        #st.markdown(
    f'<div style="text-align: center;">{pass_table.to_html(index=False)}</div>',
    unsafe_allow_html=True
)
        

