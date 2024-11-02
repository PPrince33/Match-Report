import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsbombpy import sb
from mplsoccer import Pitch

# Load competitions data
comp_df = sb.competitions()
comp_df['Tournament'] = comp_df.competition_name + ' ' + comp_df.season_name
comp_list = comp_df.Tournament.unique().tolist()

# Streamlit title
st.title("Match Report by Precious")
comp = st.selectbox("Select Competition:", comp_list)

if comp:
    # Get competition and season IDs
    c_id = comp_df.loc[comp_df['Tournament'] == comp, 'competition_id'].values[0]
    s_id = comp_df.loc[comp_df['Tournament'] == comp, 'season_id'].values[0]
    matches_df = sb.matches(competition_id=c_id, season_id=s_id)
    
    # Create match labels
    matches_df['Match'] = matches_df['home_team'] + '(' + matches_df['home_score'].astype(str) + ')' + ' vs ' + matches_df['away_team'] + '(' + matches_df['away_score'].astype(str) + ')'
    match_list = matches_df.Match.unique().tolist()
    match = st.selectbox("Select Match:", match_list)

    if match:
        match_id = matches_df.loc[matches_df['Match'] == match, 'match_id'].values[0]
        event_df = sb.events(match_id=int(match_id))
        
        # Process event data
        event_df[['X', 'Y']] = event_df['location'].apply(pd.Series)
        event_df[['endX', 'endY']] = event_df['pass_end_location'].apply(pd.Series)
        event_df['Part_of_pitch'] = np.where((event_df['X'] >= 80) & (event_df['endX'] >= 80), 'Attacking-3rd', 
                                              np.where((event_df['X'] <= 40) & (event_df['endX'] <= 40), 'Defensive-3rd', 'Mid-field'))
        
        pitch = Pitch(pitch_type='statsbomb', pitch_color='black', line_color='white')
        fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)
        fig.set_facecolor("black")
        
        st.header("Starting XI")
        team_name0 = event_df.at[0, 'team']    
        team_name1 = event_df.at[1, 'team']
        
        # Load lineups
        lineup_data0 = sb.lineups(match_id=int(match_id))[team_name0]
        lineup_data0 = lineup_data0[['player_name', 'player_id', 'jersey_number']]
        jersey_data0 = lineup_data0.copy()
        
        lineup_data1 = sb.lineups(match_id=int(match_id))[team_name1]
        lineup_data1 = lineup_data1[['player_name', 'player_id', 'jersey_number']]
        jersey_data1 = lineup_data1.copy()





        # Assuming event_df is your DataFrame and you're accessing the tactics' lineup
        lineup0 = event_df.at[0, 'tactics']['lineup']
        # List to hold extracted player information
        player_info0 = []
        # Iterate over each player in the lineup
        for player in lineup0:
            player_details0 = {
                'name': player['player']['name'],
                'position': player['position']['name'],
                'jersey_number': player['jersey_number']
            }
            player_info0.append(player_details0)
        # Create a DataFrame from the player information
        startingXI0 = pd.DataFrame(player_info0)

        # Assuming event_df is your DataFrame and you're accessing the tactics' lineup
        lineup1 = event_df.at[1, 'tactics']['lineup']
        # List to hold extracted player information
        player_info1 = []
        # Iterate over each player in the lineup
        for player in lineup1:
            player_details1 = {
                'name': player['player']['name'],
                'position': player['position']['name'],
                'jersey_number': player['jersey_number']
            }
            player_info1.append(player_details1)
        # Create a DataFrame from the player information
        startingXI1 = pd.DataFrame(player_info1)



        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{team_name0}")
            st.dataframe(startingXI0)
        with col2:
            st.subheader(f"{team_name1}")
            st.dataframe(startingXI1)
        
        # Pass data
        pass_df0 = event_df[(event_df.type == 'Pass') & (event_df.team == team_name0)]
        pass_df1 = event_df[(event_df.type == 'Pass') & (event_df.team == team_name1)]
        pass_table=pd.DataFrame(columns=[team_name0,team_name1])
        # Filter completed passes where 'pass_outcome' is null
        completed_passes_team0 = pass_df0[pass_df0['pass_outcome'].isnull()]
        completed_passes_team1 = pass_df1[pass_df1['pass_outcome'].isnull()]

        incompleted_passes_team0=pass_df0[pass_df0['pass_outcome'].notna()]
        incompleted_passes_team1=pass_df1[pass_df1['pass_outcome'].notna()]
        
        
        # Merge successful passes with lineup data
        successful0 = pd.merge(completed_passes_team0, jersey_data0, on='player_id', how='left')
        successful1 = pd.merge(completed_passes_team1, jersey_data1, on='player_id', how='left')

        # First Substitution
        subs0 = event_df[(event_df['type'] == 'Substitution') & (event_df['team'] == team_name0)]
        sub_players0 = subs0[['minute', 'player', 'player_id']]
        sub_players0 = pd.merge(sub_players0, jersey_data0, on='player_id', how='left')
        sub_players0 = sub_players0[['minute', 'player', 'jersey_number']]
        #st.subheader(f"{team_name0} Substitutions")
        #st.dataframe(sub_players0)
        firstsub0 = sub_players0[sub_players0['minute'].notnull()]['minute'].min() if not sub_players0.empty else None
        
        subs1 = event_df[(event_df['type'] == 'Substitution') & (event_df['team'] == team_name1)]
        sub_players1 = subs1[['minute', 'player', 'player_id']]
        sub_players1 = pd.merge(sub_players1, jersey_data1, on='player_id', how='left')
        sub_players1 = sub_players1[['minute', 'player', 'jersey_number']]
        #st.subheader(f"{team_name1} Substitutions")
        #st.dataframe(sub_players1)
        firstsub1 = sub_players1[sub_players1['minute'].notnull()]['minute'].min() if not sub_players1.empty else None
        col1, col2 = st.columns(2)
        with col1:
            #st.subheader(f"{team_name0} Substitutions")
            st.dataframe(sub_players0,height=200)
        with col2:
            #st.subheader(f"{team_name1} Substitutions")
            st.dataframe(sub_players1,height=200)
        # Pass Network Before First Substitution
        if firstsub0 is not None:
            successful0 = successful0[successful0['minute'] < firstsub0]
        if firstsub1 is not None:
            successful1 = successful1[successful1['minute'] < firstsub1]

        successful0.rename(columns={'player_id': 'passer_id', 'player_name': 'passer_name', 'jersey_number': 'passer_jersey_no'}, inplace=True)
        successful1.rename(columns={'player_id': 'passer_id', 'player_name': 'passer_name', 'jersey_number': 'passer_jersey_no'}, inplace=True)
        
        jersey_data0.rename(columns={'player_id': 'pass_recipient_id'}, inplace=True)
        jersey_data1.rename(columns={'player_id': 'pass_recipient_id'}, inplace=True)
        
        successful0 = pd.merge(successful0, jersey_data0, on='pass_recipient_id',how='left')
        successful1 = pd.merge(successful1, jersey_data1, on='pass_recipient_id',how='left')

        successful0.rename(columns={'player_name': 'recipient_name', 'jersey_number': 'recipient_jersey_no'}, inplace=True)
        successful1.rename(columns={'player_name': 'recipient_name', 'jersey_number': 'recipient_jersey_no'}, inplace=True)

        avg_locations0 = successful0.groupby('passer_jersey_no').agg({'X': ['mean'], 'Y': ['mean', 'count']})
        avg_locations0.columns = ['X', 'Y', 'count']
        avg_locations0.reset_index(inplace=True)
        
        avg_locations1 = successful1.groupby('passer_jersey_no').agg({'X': ['mean'], 'Y': ['mean', 'count']})
        avg_locations1.columns = ['X', 'Y', 'count']
        avg_locations1.reset_index(inplace=True)
        
        # Passes Between Players for Plotting
        pass_between0 = successful0.groupby(['passer_jersey_no', 'recipient_jersey_no']).id.count().reset_index()
        pass_between0.rename(columns={'id': 'pass_count'}, inplace=True)

        pass_between0=pd.merge(pass_between0,avg_locations0,on='passer_jersey_no',how='left')
        avg_locations0=avg_locations0.rename(columns={'passer_jersey_no':'recipient_jersey_no'})
        pass_between0=pd.merge(pass_between0,avg_locations0,on='recipient_jersey_no',suffixes=['','_end'],how='left')
        avg_locations0=avg_locations0.rename(columns={'recipient_jersey_no':'passer_jersey_no'})
        
        pass_between1 = successful1.groupby(['passer_jersey_no', 'recipient_jersey_no']).id.count().reset_index()
        pass_between1.rename(columns={'id': 'pass_count'}, inplace=True)
        
        pass_between1=pd.merge(pass_between1,avg_locations1,on='passer_jersey_no',how='left')
        avg_locations1=avg_locations1.rename(columns={'passer_jersey_no':'recipient_jersey_no'})
        pass_between1=pd.merge(pass_between1,avg_locations1,on='recipient_jersey_no',suffixes=['','_end'],how='left')
        avg_locations1=avg_locations1.rename(columns={'recipient_jersey_no':'passer_jersey_no'})

        st.subheader("Pass Analysis")
        # Set up the pitch for Team 0
        pitch0 = Pitch(pitch_type='statsbomb', pitch_color='black', line_color='#A9A9A9')
        fig0, ax0 = pitch0.draw(figsize=(8, 11), constrained_layout=True, tight_layout=False)
        fig0.set_facecolor("black")
        
        # Plot the passing lines for Team 0
        pass_lines0 = pitch0.lines(pass_between0['X'], pass_between0['Y'],
                                   pass_between0['X_end'], pass_between0['Y_end'],
                                   lw= pass_between0['pass_count'],
                                   color="white", zorder=0.7, ax=ax0)

        
        # Plot the average locations for Team 0
        pass_nodes0 = pitch0.scatter(avg_locations0['X'], avg_locations0['Y'],
                                      s=30 * avg_locations0['count'].values,
                                      color='#19AE47', edgecolors='green', linewidth=1, ax=ax0)
        
        # Annotate the plot for Team 0
        for _, row in avg_locations0.iterrows():
            pitch0.annotate(int(row['passer_jersey_no']), xy=(row['X'], row['Y']), c='black',
                    fontweight='bold', va='center', ha='center', size=10, ax=ax0)
            
        
        
        #ax0.set_title(f'{team_name0} Passing Network', color='white', va='center', ha='center',fontsize=20, fontweight='bold', pad=20)
        
        # Set up the pitch for Team 1 (similar process as for Team 0)
        pitch1 = Pitch(pitch_type='statsbomb', pitch_color='black', line_color='#A9A9A9')
        fig1, ax1 = pitch1.draw(figsize=(8, 11), constrained_layout=True, tight_layout=False)
        fig1.set_facecolor("black")
        
        # Plot the passing lines for Team 1
        pass_lines1 = pitch1.lines(pass_between1['X'], pass_between1['Y'],
                                   pass_between1['X_end'], pass_between1['Y_end'],
                                   lw=pass_between1['pass_count'],
                                   color="white", zorder=0.7, ax=ax1)
        
        # Plot the average locations for Team 1
        pass_nodes1 = pitch1.scatter(avg_locations1['X'], avg_locations1['Y'],
                                      s=30 * avg_locations1['count'].values,
                                      color='#19AE47', edgecolors='green', linewidth=1, ax=ax1)
        
        for _, row in avg_locations1.iterrows():
            pitch1.annotate(int(row['passer_jersey_no']), xy=(row['X'], row['Y']), c='black',
                    fontweight='bold', va='center', ha='center', size=10, ax=ax1)

        
        
        #ax1.set_title(f'{team_name1} Passing Network', color='white', va='center', ha='center',fontsize=20, fontweight='bold', pad=20)
        # Display the plots
        
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig0)
        with col2:
            st.pyplot(fig1)
        
        
        
        
        
        
        
        
        
       
        
        pitch = Pitch(pitch_type='statsbomb', pitch_color='black', line_color='white')
        fig, ax = pitch.draw(figsize=(16, 11), constrained_layout=True, tight_layout=False)
        fig.set_facecolor("black")
        # Filter passes where 'pass_outcome' is null for both DataFrames
        completed_passes_team0 = pass_df0[pass_df0['pass_outcome'].isnull()]
        completed_passes_team1 = pass_df1[pass_df1['pass_outcome'].isnull()]       
        
                
                
        # Unique options for Part_of_pitch and player for both DataFrames
        part_of_pitch_options0 = ['All'] + completed_passes_team0['Part_of_pitch'].unique().tolist()
        part_of_pitch_options1 = ['All'] + completed_passes_team1['Part_of_pitch'].unique().tolist()
        player_options0 = ['All'] + completed_passes_team0['player'].unique().tolist()
        player_options1 = ['All'] + completed_passes_team1['player'].unique().tolist()
        pass_type0=['All'] + completed_passes_team0.pass_type.unique().tolist()
        pass_type1=['All'] + completed_passes_team1.pass_type.unique().tolist()
        # Team 1 filters (completed_passes_team1)
        st.sidebar.subheader(f"{team_name0} Filters")
        part_of_pitch_selected0 = st.sidebar.selectbox(f"Select Part of Pitch {team_name0}", options=part_of_pitch_options0)
        players_selected0 = st.sidebar.selectbox(f"Select Player(s) {team_name0}", options=player_options0)
        pass_type_selected0 = st.sidebar.selectbox(f"Select Pass Type {team_name0}", options=pass_type0)
        minute_slider0 = st.sidebar.slider(f"Select Minute Range {team_name0}", min_value=int(completed_passes_team0['minute'].min()), max_value=int(completed_passes_team0['minute'].max()), value=(int(completed_passes_team0['minute'].min()), int(completed_passes_team0['minute'].max())))
        
        # Team 2 filters (completed_passes_team2)
        st.sidebar.subheader(f"{team_name1} Filters")
        part_of_pitch_selected1 = st.sidebar.selectbox(f"Select Part of Pitch {team_name1}", options=part_of_pitch_options1)
        players_selected1 = st.sidebar.selectbox(f"Select Player(s) {team_name1}", options=player_options1)
        pass_type_selected1 = st.sidebar.selectbox(f"Select Pass Type {team_name1}", options=pass_type1)
        minute_slider1 = st.sidebar.slider(f"Select Minute Range {team_name1}", min_value=int(completed_passes_team1['minute'].min()), max_value=int(completed_passes_team1['minute'].max()), value=(int(completed_passes_team1['minute'].min()), int(completed_passes_team1['minute'].max())))
        
        # Apply filters to completed_passes_team1 based on selected Part of Pitch and Players
        if part_of_pitch_selected0 != 'All':
            completed_passes_team0 = completed_passes_team0[completed_passes_team0['Part_of_pitch'] == part_of_pitch_selected0]
            incompleted_passes_team0=incompleted_passes_team0[incompleted_passes_team0['Part_of_pitch'] == part_of_pitch_selected0]
            

        if players_selected0 != 'All':
            completed_passes_team0 = completed_passes_team0[completed_passes_team0['player'] == players_selected0]
            incompleted_passes_team0=incompleted_passes_team0[incompleted_passes_team0['player'] == players_selected0]
                  
        # Filter based on minute range for Team 1
        completed_passes_team0 = completed_passes_team0[(completed_passes_team0['minute'] >= minute_slider0[0]) & (completed_passes_team0['minute'] <= minute_slider0[1])]
        incompleted_passes_team0 = incompleted_passes_team0[(incompleted_passes_team0['minute'] >= minute_slider0[0]) & (incompleted_passes_team0['minute'] <= minute_slider0[1])]
        
        # Apply filters to completed_passes_team2 based on selected Part of Pitch and Players
        if part_of_pitch_selected1 != 'All':
            completed_passes_team1 = completed_passes_team1[completed_passes_team1['Part_of_pitch'] == part_of_pitch_selected1]
            incompleted_passes_team1=incompleted_passes_team1[incompleted_passes_team1['Part_of_pitch'] == part_of_pitch_selected1]
            
        if players_selected1 != 'All':
            completed_passes_team1 = completed_passes_team1[completed_passes_team1['player'] == players_selected1]
            incompleted_passes_team1=incompleted_passes_team1[incompleted_passes_team1['player'] == part_of_pitch_selected1]
            
        if pass_type_selected0 != 'All':
            completed_passes_team0 = completed_passes_team0[completed_passes_team0['pass_type'] == pass_type_selected0]
            incompleted_passes_team0=incompleted_passes_team0[incompleted_passes_team0['pass_type'] == pass_type_selected0]
            
        if pass_type_selected1 != 'All':
            completed_passes_team1 = completed_passes_team1[completed_passes_team1['pass_type'] == pass_type_selected1]
            incompleted_passes_team1=incompleted_passes_team1[incompleted_passes_team1['pass_type'] == pass_type_selected1]
            
        # Filter based on minute range for Team 2
        completed_passes_team1 = completed_passes_team1[(completed_passes_team1['minute'] >= minute_slider1[0]) & (completed_passes_team1['minute'] <= minute_slider1[1])]
        incompleted_passes_team1 = incompleted_passes_team1[(incompleted_passes_team1['minute'] >= minute_slider1[0]) & (incompleted_passes_team1['minute'] <= minute_slider1[1])]
        
        # Initialize the pitch settings
        pitch = Pitch(pitch_type='statsbomb', pitch_color='black', line_color='white')
        
        # Streamlit layout for side-by-side pitch maps
        st.subheader("Successful Passes")
        col1, col2 = st.columns(2)
        
        # Plot for Team 1 with filters applied
        with col1:
            fig, ax = pitch.draw(figsize=(8, 6), constrained_layout=True, tight_layout=False)
            fig.set_facecolor("black")
            # Draw lines for each pass
            ax.scatter(completed_passes_team0['X'], completed_passes_team0['Y'], color='green', label="Start")
            ax.scatter(completed_passes_team0['endX'], completed_passes_team0['endY'], color='red', label="End")
            
            for idx, row in completed_passes_team0.iterrows():
                ax.plot([row['X'], row['endX']], [row['Y'], row['endY']], color='white',linestyle='--', linewidth=1)  # Line between passes
            ax.legend(loc="upper left")
            st.pyplot(fig)
        
        # Plot for Team 2 with filters applied
        with col2:
            fig, ax = pitch.draw(figsize=(8, 6), constrained_layout=True, tight_layout=False)
            fig.set_facecolor("black")
            # Draw lines for each pass
            ax.scatter(completed_passes_team1['X'], completed_passes_team1['Y'], color='green', label="Start")
            ax.scatter(completed_passes_team1['endX'], completed_passes_team1['endY'], color='red', label="End")
            
            for idx, row in completed_passes_team1.iterrows():
                ax.plot([row['X'], row['endX']], [row['Y'], row['endY']], color='white', linestyle='--',linewidth=1)  # Line between passes
            ax.legend(loc="upper left")
            st.pyplot(fig)
        
        
 
          #Create an empty DataFrame for pass tables
        completed_pass_table = pd.DataFrame()
        incompleted_pass_table = pd.DataFrame()
        
        # Completed Pass counts
        completed_pass_count0 = pass_df0[pass_df0['pass_outcome'].isnull()].id.count()
        completed_pass_count1 = pass_df1[pass_df1['pass_outcome'].isnull()].id.count()
        completed_pass_table.loc['Completed Pass', team_name0] = completed_pass_count0
        completed_pass_table.loc['Completed Pass', team_name1] = completed_pass_count1
        
        # Pass type counts for completed passes
        pass_type_counts0 = pass_df0['pass_type'].value_counts()
        for pass_type, count in pass_type_counts0.items():
            completed_pass_table.loc[pass_type, team_name0] = count
        
        pass_type_counts1 = pass_df1['pass_type'].value_counts()
        for pass_type, count in pass_type_counts1.items():
            completed_pass_table.loc[pass_type, team_name1] = count
        
        # Pass outcome counts for completed passes
        pass_outcome_counts0 = pass_df0['pass_outcome'].value_counts()
        for outcome, count in pass_outcome_counts0.items():
            completed_pass_table.loc[outcome, team_name0] = count
        
        pass_outcome_counts1 = pass_df1['pass_outcome'].value_counts()
        for outcome, count in pass_outcome_counts1.items():
            completed_pass_table.loc[outcome, team_name1] = count
        
        # Fill NaN values in completed pass table
        completed_pass_table = completed_pass_table.fillna(0)
        
        # Reset index for completed pass table
        completed_pass_table = completed_pass_table.reset_index().rename(columns={'index': 'Particular'})
        
        # Incompleted Pass counts
        incompleted_pass_count0 = pass_df0[pass_df0['pass_outcome'].notnull()].id.count()
        incompleted_pass_count1 = pass_df1[pass_df1['pass_outcome'].notnull()].id.count()
        
        incompleted_pass_table.loc['Incompleted Pass', team_name0] = incompleted_pass_count0
        incompleted_pass_table.loc['Incompleted Pass', team_name1] = incompleted_pass_count1
        
        # Pass type counts for incomplete passes
        inpass_type_counts0 = pass_df0[pass_df0['pass_outcome'].notnull()]['pass_type'].value_counts()
        for pass_type, count in inpass_type_counts0.items():
            incompleted_pass_table.loc[pass_type, team_name0] = count
        
        inpass_type_counts1 = pass_df1[pass_df1['pass_outcome'].notnull()]['pass_type'].value_counts()
        for pass_type, count in inpass_type_counts1.items():
            incompleted_pass_table.loc[pass_type, team_name1] = count
        
        # Pass outcome counts for incomplete passes
        inpass_outcome_counts0 = pass_df0[pass_df0['pass_outcome'].notnull()]['pass_outcome'].value_counts()
        for outcome, count in inpass_outcome_counts0.items():
            incompleted_pass_table.loc[outcome, team_name0] = count
        
        inpass_outcome_counts1 = pass_df1[pass_df1['pass_outcome'].notnull()]['pass_outcome'].value_counts()
        for outcome, count in inpass_outcome_counts1.items():
            incompleted_pass_table.loc[outcome, team_name1] = count
        
        # Fill NaN values in incomplete pass table
        incompleted_pass_table = incompleted_pass_table.fillna(0)
        
        # Reset index for incomplete pass table
        incompleted_pass_table = incompleted_pass_table.reset_index().rename(columns={'index': 'Particular'})
        

        
        
        
                
        
                
        # Initialize the pitch settings
        pitch = Pitch(pitch_type='statsbomb', pitch_color='black', line_color='white')
        
        # Streamlit layout for side-by-side pitch maps
        st.subheader("Unsuccessful Passes")
        col1, col2 = st.columns(2)


        
        # Plot for Team 1 with filters applied
        with col1:
            fig, ax = pitch.draw(figsize=(8, 6), constrained_layout=True, tight_layout=False)
            fig.set_facecolor("black")
            # Draw lines for each pass
            ax.scatter(incompleted_passes_team0['X'], incompleted_passes_team0['Y'], color='green', label="Start")
            ax.scatter(incompleted_passes_team0['endX'], incompleted_passes_team0['endY'], color='red', label="End")
            
            for idx, row in incompleted_passes_team0.iterrows():
                ax.plot([row['X'], row['endX']], [row['Y'], row['endY']], color='white',linestyle='--', linewidth=1)  # Line between passes
            ax.legend(loc="upper left")
            st.pyplot(fig)
        
        # Plot for Team 2 with filters applied
        with col2:
            fig, ax = pitch.draw(figsize=(8, 6), constrained_layout=True, tight_layout=False)
            fig.set_facecolor("black")
            # Draw lines for each pass
            ax.scatter(incompleted_passes_team1['X'], incompleted_passes_team1['Y'], color='green', label="Start")
            ax.scatter(incompleted_passes_team1['endX'], incompleted_passes_team1['endY'], color='red', label="End")
            
            for idx, row in incompleted_passes_team1.iterrows():
                ax.plot([row['X'], row['endX']], [row['Y'], row['endY']], color='white', linestyle='--',linewidth=1)  # Line between passes
            ax.legend(loc="upper left")
            st.pyplot(fig)



        # Display the completed pass table
        st.subheader("Completed Pass Table")
        st.dataframe(completed_pass_table)
        
        # Display the incomplete pass table
        st.subheader("Incomplete Pass Table")
        st.dataframe(incompleted_pass_table)

        #st.markdown(f"<div style='display: flex; justify-content: center;'>{pass_table.to_html(index=False)}</div>",unsafe_allow_html=True)
