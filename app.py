import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsbombpy import sb
from mplsoccer import Pitch
import streamlit as st

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
        selected_match_id = matches_df.loc[matches_df['Match'] == match, 'match_id'].values[0]
        event_df = sb.events(match_id=int(selected_match_id))
        
        event_df=sb.events(match_id=selected_match_id)
        location_columns = ['shot_end_location', 'goalkeeper_end_location', 'carry_end_location', 'location', 'pass_end_location']
        for col in location_columns:
            # Extract x, y, z as new columns for each location type, handling missing values
            event_df[f'{col}_x'] = event_df[col].apply(lambda loc: loc[0] if isinstance(loc, (list, tuple)) and len(loc) > 0 else None)
            event_df[f'{col}_y'] = event_df[col].apply(lambda loc: loc[1] if isinstance(loc, (list, tuple)) and len(loc) > 1 else None)
            event_df[f'{col}_z'] = event_df[col].apply(lambda loc: loc[2] if isinstance(loc, (list, tuple)) and len(loc) > 2 else None)


        event_df['Part_of_pitch'] = np.where((event_df['location_x'] >= 80), 'Attacking-3rd', 
                                              np.where((event_df['location_x'] <= 40), 'Defensive-3rd', 'Mid-field'))

        
        #Starting Lineup Data Frame
        S_lineup_df=event_df[event_df.type=='Starting XI'].dropna(how='all',axis=1).reset_index(drop=True)
        
        #Team Names
        team_name0=S_lineup_df.at[0,'team']
        team_name1=S_lineup_df.at[1,'team']
        
        #Formation
        formation0=S_lineup_df.at[0,'tactics']['formation']
        formation1=S_lineup_df.at[1,'tactics']['formation']
        
        #Starting Lineup
        #Substitutions
        full_lineup0=sb.lineups(match_id=selected_match_id)[team_name0]
        positions_expanded0 = full_lineup0['positions'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else {}).apply(pd.Series)
        card_expanded0=full_lineup0['cards'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else {}).apply(pd.Series)
        full_lineup_expanded0 = pd.concat([full_lineup0, positions_expanded0], axis=1)
        full_lineup_expanded0 = pd.concat([full_lineup_expanded0, card_expanded0], axis=1)
        full_lineup_expanded0 = full_lineup_expanded0.drop(columns=['positions','cards'])
        full_lineup_expanded0.sort_values(by=['start_reason','end_reason'],inplace=True)
        if 'card_type'in full_lineup_expanded0.columns:
            full_lineup_expanded0=full_lineup_expanded0[['player_name','jersey_number','position','from',
                                                     'to','start_reason','end_reason','card_type','time','reason']].reset_index(drop=True)
        else:
            full_lineup_expanded0=full_lineup_expanded0[['player_name','jersey_number','position','from',
                                                     'to','start_reason','end_reason']].reset_index(drop=True)
        ##----------------------------------------------------------|
        full_lineup1=sb.lineups(match_id=selected_match_id)[team_name1]
        positions_expanded1 = full_lineup1['positions'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else {}).apply(pd.Series)
        card_expanded1=full_lineup1['cards'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else {}).apply(pd.Series)
        full_lineup_expanded1 = pd.concat([full_lineup1, positions_expanded1], axis=1)
        full_lineup_expanded1 = pd.concat([full_lineup_expanded1, card_expanded1], axis=1)
        full_lineup_expanded1 = full_lineup_expanded1.drop(columns=['positions','cards'])
        full_lineup_expanded1.sort_values(by=['start_reason','end_reason'],inplace=True)
        if 'card_type'in full_lineup_expanded1.columns:
            full_lineup_expanded1=full_lineup_expanded1[['player_name','jersey_number','position','from',
                                                     'to','start_reason','end_reason','card_type','time','reason']].reset_index(drop=True)
        else:
            full_lineup_expanded1=full_lineup_expanded1[['player_name','jersey_number','position','from',
                                                     'to','start_reason','end_reason']].reset_index(drop=True)
        ##----------------------------------------------------------|
        startingXI0 = full_lineup_expanded0[full_lineup_expanded0.start_reason=='Starting XI'].rename(columns={'player_name':'player'})
        startingXI1 = full_lineup_expanded1[full_lineup_expanded1.start_reason=='Starting XI'].rename(columns={'player_name':'player'})
        
        sub0=full_lineup_expanded0[full_lineup_expanded0.end_reason.isin(['Substitution - Off (Injury)','Substitution - Off (Tactical)'])].rename(columns={'player_name':'player'})
        sub1=full_lineup_expanded1[full_lineup_expanded1.end_reason.isin(['Substitution - Off (Injury)','Substitution - Off (Tactical)'])].rename(columns={'player_name':'player'})
        
        
        
        poss0 = event_df[event_df.possession_team == team_name0]['possession_team'].count() / event_df.possession_team.count()
        poss1 = event_df[event_df.possession_team == team_name1]['possession_team'].count() / event_df.possession_team.count()
        fig, ax = plt.subplots(figsize=(18, 1))
        ax.barh(y=0, width=poss0,color='#DEEFF5', label=f'{team_name0} - {poss0 * 100:.1f}%')
        ax.barh(y=0, width=poss1, left=poss0, color='#90EE90', label=f'{team_name1} - {poss1 * 100:.1f}%')
        ax.axis('off')
        ax.text(poss0 / 2, 0, f"{team_name0} - {poss0 * 100:.1f}%",fontsize=14, ha='center', va='center', fontweight='bold', color='black',fontname="Georgia")
        ax.text(poss0 + poss1 / 2, 0, f"{team_name1} - {poss1 * 100:.1f}%",fontsize=14, ha='center', va='center', fontweight='bold', color='black',fontname="Georgia")
        plt.suptitle("Possession", fontsize=14, fontweight='bold',fontname="Georgia",y=1.1)
        
        st.pyplot(fig)
        
        
        all_avg_loc0=event_df[event_df.team==team_name0].groupby('player')[['location_x','location_y']].mean().reset_index()
        all_avg_loc0=pd.merge(startingXI0,all_avg_loc0,on='player',how='left')
        all_avg_loc1=event_df[event_df.team==team_name1].groupby('player')[['location_x','location_y']].mean().reset_index()
        all_avg_loc1=pd.merge(startingXI1,all_avg_loc1,on='player',how='left')
        # Formation Pitch
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 10), constrained_layout=True)
        plt.suptitle("Formation", fontsize=14, fontweight='bold',fontname="Georgia",y=0.8)
        # Draw pitch and plot for team 0
        pitch0 = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
        pitch0.draw(ax=ax0)
        for i in range(all_avg_loc0.shape[0]):
            ax0.scatter(all_avg_loc0['location_x'][i], all_avg_loc0['location_y'][i], color='#DEEFF5', edgecolors='black', s=800)
            ax0.text(all_avg_loc0['location_x'][i], all_avg_loc0['location_y'][i], s=all_avg_loc0['jersey_number'][i],
                     color='black', weight='bold', ha='center', va='center',fontsize=15,fontname="Georgia", zorder=2)
        ax0.set_title(f'{team_name0} - {formation0}', fontsize=14, fontweight='bold', fontname="Georgia", y=0.97)
        
        # Draw pitch and plot for team 1
        pitch1 = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
        pitch1.draw(ax=ax1)
        for i in range(all_avg_loc1.shape[0]):
            ax1.scatter(all_avg_loc1['location_x'][i], all_avg_loc1['location_y'][i], color='#90EE90', edgecolors='black', s=800)
            ax1.text(all_avg_loc1['location_x'][i], all_avg_loc1['location_y'][i], s=all_avg_loc1['jersey_number'][i],
                     color='black', weight='bold', ha='center', va='center',fontsize=15,fontname="Georgia", zorder=2)
        ax1.set_title(f'{team_name1} - {formation1}', fontsize=14, fontweight='bold', fontname="Georgia", y=0.97)
        
        st.pyplot(fig)
        ol1, col2 = st.columns(2)

        # Display the DataFrames in the columns
        with col1:
            st.subheader(f"{team_name0} Starting XI")
            st.dataframe(startingXI0)
            st.subheader(f"Substitutions")
            st.dataframe(sub0)
            pitch0 = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
            fig, ax0 = plt.subplots(figsize=(10, 6))
            pitch0.draw(ax=ax0)
            for i in range(all_avg_loc0.shape[0]):
                pitch0.scatter(all_avg_loc0['location_x'][i], all_avg_loc0['location_y'][i], 
                               color='#DEEFF5', edgecolors='black', s=600, ax=ax0)
                pitch0.text(all_avg_loc0['location_x'][i], all_avg_loc0['location_y'][i], 
                            s=all_avg_loc0['jersey_number'][i], color='black', weight='bold', 
                            ha='center', va='center', fontsize=15, fontname="Georgia", zorder=2, ax=ax0)
            ax0.set_title(f'{team_name0} - {formation0}', fontsize=14, fontweight='bold', fontname="Georgia", y=0.97)
            plt.show()
        
        with col2:
            st.subheader(f"{team_name0} Starting XI")
            st.dataframe(startingXI1)
            st.subheader(f"Substitutions")
            st.dataframe(sub1)               
            pitch1 = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
            fig, ax1 = plt.subplots(figsize=(10, 6))
            pitch1.draw(ax=ax0)
            for i in range(all_avg_loc1.shape[0]):
                pitch1.scatter(all_avg_loc1['location_x'][i], all_avg_loc1['location_y'][i], 
                               color='#DEEFF5', edgecolors='black', s=600, ax=ax1)
                pitch1.text(all_avg_loc1['location_x'][i], all_avg_loc1['location_y'][i], 
                            s=all_avg_loc1['jersey_number'][i], color='black', weight='bold', 
                            ha='center', va='center', fontsize=15, fontname="Georgia", zorder=2, ax=ax1)
            ax1.set_title(f'{team_name1} - {formation1}', fontsize=14, fontweight='bold', fontname="Georgia", y=0.97)
            plt.show()       
                    
                    
                    
                    
                    
                    
                    
                    
