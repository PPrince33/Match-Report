import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
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
        
        sub0=full_lineup_expanded0[full_lineup_expanded0.start_reason.isin(['Substitution - On (Injury)','Substitution - On (Tactical)'])].rename(columns={'player_name':'player'})
        sub1=full_lineup_expanded1[full_lineup_expanded1.start_reason.isin(['Substitution - On (Injury)','Substitution - On (Tactical)'])].rename(columns={'player_name':'player'})
        
        
        
        poss0 = event_df[(event_df.possession_team==team_name0)]['possession_team'].count() / event_df.possession_team.count()
        poss1 = event_df[(event_df.possession_team==team_name1)]['possession_team'].count() / event_df.possession_team.count()
        
        
        all_avg_loc0=event_df[(event_df.team==team_name0)&(event_df.possession_team==team_name0)].groupby('player')[['location_x','location_y']].mean().reset_index()
        all_avg_loc0=pd.merge(startingXI0,all_avg_loc0,on='player',how='left')
        all_avg_loc1=event_df[(event_df.team==team_name1)&(event_df.possession_team==team_name1)].groupby('player')[['location_x','location_y']].mean().reset_index()
        all_avg_loc1=pd.merge(startingXI1,all_avg_loc1,on='player',how='left')
        fig, ax = plt.subplots(figsize=(18, 1))
        ax.barh(y=0, width=poss0, color='#DEEFF5', label=f'{team_name0} - {poss0 * 100:.1f}%')
        ax.barh(y=0, width=poss1, left=poss0, color='#90EE90', label=f'{team_name1} - {poss1 * 100:.1f}%')
        ax.axis('off')
        ax.text(poss0 / 2, 0, f"{team_name0} - {poss0 * 100:.1f}%",fontsize=14, ha='center', va='center', fontweight='bold', color='black',fontname="Georgia")
        ax.text(poss0 + poss1 / 2, 0, f"{team_name1} - {poss1 * 100:.1f}%",fontsize=14, ha='center', va='center', fontweight='bold', color='black',fontname="Georgia")
        plt.suptitle("Possession", fontsize=14, fontweight='bold',fontname="Georgia",y=1.1)
        st.pyplot(fig) 

        if 'card_type'in startingXI0.columns:
            startingXI0=startingXI0[['player','jersey_number','position','to','card_type','time','reason']]
        else:
             startingXI0=startingXI0[['player','jersey_number','position','to']]
        
        if 'card_type'in startingXI1.columns:
            startingXI1=startingXI1[['player','jersey_number','position','to','card_type','time','reason']]
        else:
            startingXI1=startingXI1[['player','jersey_number','position','to']]
        
        startingXI0.rename(columns={'player':'Player','jersey_number':'Jersey No.','position':'Position','to':'Sub Out Time','card_type':'Card','time':'Card Time','reason':'Card Reason'},inplace=True)
        startingXI1.rename(columns={'player':'Player','jersey_number':'Jersey No.','position':'Position','to':'Sub Out Time','card_type':'Card','time':'Card Time','reason':'Card Reason'},inplace=True)
        
        if 'card_type'in sub0.columns:
            sub0=sub0[['player','jersey_number','position','from','to','card_type','time','reason']]
        else:
            sub0=sub0[['player','jersey_number','position','from','to']]
        
        if 'card_type'in sub1.columns:
            sub1=sub1[['player','jersey_number','position','from','to','card_type','time','reason']]
        else:
            sub1=sub1[['player','jersey_number','position','from','to']]
    

        
        sub0.rename(columns={'player':'Player','jersey_number':'Jersey No.','position':'Position','from':'Sub In Time','to':'Sub Out Time','card_type':'Card','time':'Card Time','reason':'Card Reason'},inplace=True)
        sub1.rename(columns={'player':'Player','jersey_number':'Jersey No.','position':'Position','from':'Sub In Time','to':'Sub Out Time','card_type':'Card','time':'Card Time','reason':'Card Reason'},inplace=True)

        
        sub_min=event_df[event_df.type=='Substitution'][['team','player','minute']].groupby('team')['minute'].min().reset_index()
        sub_min0=int(sub_min[sub_min.team==team_name0]['minute'])
        sub_min1=int(sub_min[sub_min.team==team_name1]['minute'])
        
        
        pass_df0_net=event_df[(event_df.type=='Pass')&(event_df.team==team_name0)&(event_df.pass_outcome.isnull())&(event_df.minute<sub_min0)].dropna(axis=1,how='all')
        pass_df1_net=event_df[(event_df.type=='Pass')&(event_df.team==team_name1)&(event_df.pass_outcome.isnull())&(event_df.minute<sub_min1)].dropna(axis=1,how='all')
        
        pass_df0_avg_loc=pass_df0_net.groupby('player')[['location_x','location_y']].mean().reset_index()
        pass_df1_avg_loc=pass_df1_net.groupby('player')[['location_x','location_y']].mean().reset_index()
        
        pass_bw0=pass_df0_net.groupby(['player','pass_recipient'])['id'].count().reset_index()
        pass_bw1=pass_df1_net.groupby(['player','pass_recipient'])['id'].count().reset_index()
        
        pass_bw0=pd.merge(pass_bw0,pass_df0_avg_loc,on='player')
        pass_bw1=pd.merge(pass_bw1,pass_df1_avg_loc,on='player')
        
        pass_recipient_df0_avg_loc=pass_df0_avg_loc.rename(columns={'player':'pass_recipient','location_x':'end_x','location_y':'end_y'})
        pass_recipient_df1_avg_loc=pass_df1_avg_loc.rename(columns={'player':'pass_recipient','location_x':'end_x','location_y':'end_y'})
        
        pass_bw0=pd.merge(pass_bw0,pass_recipient_df0_avg_loc,on='pass_recipient')
        pass_bw1=pd.merge(pass_bw1,pass_recipient_df1_avg_loc,on='pass_recipient')
        
        pass_df0_avg_loc.rename(columns={'player':'Player'},inplace=True)
        pass_df1_avg_loc.rename(columns={'player':'Player'},inplace=True)
        
        pass_df0_avg_loc=pd.merge(startingXI0[['Player','Jersey No.']],pass_df0_avg_loc,on='Player')
        pass_df1_avg_loc=pd.merge(startingXI1[['Player','Jersey No.']],pass_df1_avg_loc,on='Player')
        
        event_df['Pass_Result']=np.where((event_df.type=='Pass') & (event_df.pass_outcome.isnull()),1,0)
        pass_df0=event_df[(event_df.type=='Pass')&(event_df.team==team_name0)].dropna(axis=1,how='all')
        pass_df1=event_df[(event_df.type=='Pass')&(event_df.team==team_name1)].dropna(axis=1,how='all')
        pass_result0=pass_df0.groupby(['player','Pass_Result'])['id'].count().reset_index().rename(columns={'id':'counts'})
        pass_result1=pass_df1.groupby(['player','Pass_Result'])['id'].count().reset_index().rename(columns={'id':'counts'})
        pivot_pass_result0= pass_result0.pivot_table(index='player', columns='Pass_Result', values='counts', fill_value=0).reset_index()
        pivot_pass_result1 = pass_result1.pivot_table(index='player', columns='Pass_Result', values='counts', fill_value=0).reset_index()
        
        
        
        successful_pass0_number=pass_df0[pass_df0.pass_outcome.isnull()].shape[0]
        successful_pass1_number=pass_df1[pass_df1.pass_outcome.isnull()].shape[0]
        successful_pass0_rate=round(pass_df0[pass_df0.pass_outcome.isnull()].shape[0]/pass_df0.shape[0]*100)
        successful_pass1_rate=round(pass_df1[pass_df1.pass_outcome.isnull()].shape[0]/pass_df1.shape[0]*100)
        
        
        
        pass_rate_player0=pd.DataFrame(round(pass_result0[pass_result0['Pass_Result'] == 1].groupby('player')['counts'].sum()/pass_result0.groupby('player')['counts'].sum()*100,2)).reset_index()
        pass_rate_player0=pd.merge(pivot_pass_result0,pass_rate_player0,on='player')
        pass_rate_player0=pass_rate_player0.rename(columns={'player':'Player',0:'Unsucessful Passes',1:'Successful Passes','counts':'Pass Accuracy'})
        pass_rate_player0=pass_rate_player0.sort_values(by=['Successful Passes','Pass Accuracy'], ascending=False).reset_index(drop=True)
        
        
        pass_rate_player1=pd.DataFrame(round(pass_result1[pass_result1['Pass_Result'] == 1].groupby('player')['counts'].sum()/pass_result1.groupby('player')['counts'].sum()*100,2)).reset_index()
        pass_rate_player1=pd.merge(pivot_pass_result1,pass_rate_player1,on='player')
        pass_rate_player1=pass_rate_player1.rename(columns={'player':'Player',0:'Unsucessful Passes',1:'Successful Passes','counts':'Pass Accuracy'})
        pass_rate_player1=pass_rate_player1.sort_values(by=['Successful Passes','Pass Accuracy'], ascending=False).reset_index(drop=True)
        
        
        pass_df0['progressive_pass']=np.where((pass_df0.location_x<pass_df0.pass_end_location_x),1,0)
        pass_df1['progressive_pass']=np.where((pass_df1.location_x<pass_df1.pass_end_location_x),1,0)
        
        pass_df0.pass_type.fillna('Normal Pass',inplace=True)
        pass_df1.pass_type.fillna('Normal Pass',inplace=True)
        
        successful_progressive_pass0_number=pass_df0[(pass_df0['progressive_pass']==1)&pass_df0.pass_outcome.isnull()].shape[0]
        successful_progressive_pass1_number=pass_df1[(pass_df1['progressive_pass']==1)&pass_df1.pass_outcome.isnull()].shape[0]

        
        



        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        


        
    
        min_passes_options = [1, 2, 3, 4, 5,6 ,7, 8, 9, 10, 11, 12, 13]
        min_passes = st.sidebar.selectbox("Select Minimum Passes (Pass Network)", min_passes_options)

        col1, col2 = st.columns(2)
        # Display the DataFrames in the columns
        with col1:
            st.subheader(f"{team_name0} XI")
            st.dataframe(startingXI0, height=400, width=700)
            st.subheader(f"Substitutions")
            st.dataframe(sub0, height=150, width=700)
            pitch0 = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
            fig0, ax0 = plt.subplots(figsize=(10, 6))
            pitch0.draw(ax=ax0)
            for i in range(all_avg_loc0.shape[0]):
                pitch0.scatter(all_avg_loc0['location_x'][i], all_avg_loc0['location_y'][i], 
                               color='#DEEFF5', edgecolors='black', s=600, ax=ax0)
                pitch0.text(all_avg_loc0['location_x'][i], all_avg_loc0['location_y'][i], 
                            s=all_avg_loc0['jersey_number'][i], color='black', weight='bold', 
                            ha='center', va='center', fontsize=15, fontname="Georgia", zorder=2, ax=ax0)
            ax0.set_title(f'{team_name0} - {formation0}', fontsize=14, fontweight='bold', fontname="Georgia", y=0.97)
            plt.show()
            st.subheader(f"Average Formation or Position")
            st.pyplot(fig0)

            pitch00 = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
            fig00, ax00 = plt.subplots(figsize=(10, 6))
            pitch00.draw(ax=ax00)

            pass_bw0_set = pass_bw0[pass_bw0['id'] >= int(min_passes)].reset_index(drop=True)
            for i in range(pass_bw0_set.shape[0]):
                start_x = pass_bw0_set.iloc[i]['location_x']
                start_y = pass_bw0_set.iloc[i]['location_y']
                end_x = pass_bw0_set.iloc[i]['end_x']
                end_y = pass_bw0_set.iloc[i]['end_y']
                size = pass_bw0_set.iloc[i]['id']  # Adjust multiplier if lines are too thick/thin
                pitch00.lines(start_x, start_y, end_x, end_y, ax=ax00, color="black", lw=size, zorder=1)
            for i in range(pass_df0_avg_loc.shape[0]):
                ax00.scatter(pass_df0_avg_loc.iloc[i]['location_x'], pass_df0_avg_loc.iloc[i]['location_y'], 
                             color='#DEEFF5', edgecolors='black', s=600, linewidths=2)
                ax00.text(pass_df0_avg_loc.iloc[i]['location_x'], pass_df0_avg_loc.iloc[i]['location_y'], 
                          s=pass_df0_avg_loc.iloc[i]['Jersey No.'], color='black', weight='bold', 
                          ha='center', va='center', fontsize=15, fontname="Georgia", zorder=2)
            ax00.set_title(f'{team_name0} - {formation0} (0 to {sub_min0} mins)', fontsize=14, fontweight='bold', fontname="Georgia", y=0.97)
            pitch00.kdeplot(x=pass_df0.location_x,y=pass_df0.location_y,cmap="Blues",
                shade=True,
                n_levels=10,
                alpha=0.5,
                zorder=0,ax=ax00 ,
                    linewidths=0 
                
            )
                        
            
            plt.show()
            st.subheader(f"Pass Network")
            st.pyplot(fig00)
            st.write("No. of successfull passes:", successful_pass0_number)        
            st.write("No. of successfull progressive passes:", successful_progressive_pass0_number)
            st.metric(label="Pass Accuracy", value=f"{successful_pass0_rate}%", delta=None)
            st.subheader(f"Players Pass Accuracy")
            st.dataframe(pass_rate_player0)   
            

            













        
        
        with col2:
            st.subheader(f"{team_name1} XI")
            st.dataframe(startingXI1, height=400, width=700)
            st.subheader(f"Substitutions")
            st.dataframe(sub1, height=150, width=700)               
            pitch1 = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            pitch1.draw(ax=ax1)
            for i in range(all_avg_loc1.shape[0]):
                pitch1.scatter(all_avg_loc1['location_x'][i], all_avg_loc1['location_y'][i], 
                               color='#90EE90', edgecolors='black', s=600, ax=ax1)
                pitch1.text(all_avg_loc1['location_x'][i], all_avg_loc1['location_y'][i], 
                            s=all_avg_loc1['jersey_number'][i], color='black', weight='bold', 
                            ha='center', va='center', fontsize=15, fontname="Georgia", zorder=2, ax=ax1)
            ax1.set_title(f'{team_name1} - {formation1}', fontsize=14, fontweight='bold', fontname="Georgia", y=0.97)
            plt.show()       
            st.subheader(f"Average Formation or Position")
            st.pyplot(fig1)       
                    
                    
            pitch11 = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
            fig11, ax11 = plt.subplots(figsize=(10, 6))
            pitch11.draw(ax=ax11)

      
            pass_bw1_set = pass_bw1[pass_bw1['id'] >= int(min_passes)].reset_index(drop=True)
           
            for i in range(pass_bw1_set.shape[0]):
                start_x = pass_bw1_set.iloc[i]['location_x']
                start_y = pass_bw1_set.iloc[i]['location_y']
                end_x = pass_bw1_set.iloc[i]['end_x']
                end_y = pass_bw1_set.iloc[i]['end_y']
                size = pass_bw1_set.iloc[i]['id']  # Adjust multiplier if lines are too thick/thin
                pitch11.lines(start_x, start_y, end_x, end_y, ax=ax11, color="black", lw=size, zorder=1)
            for i in range(pass_df1_avg_loc.shape[0]):
                ax11.scatter(pass_df1_avg_loc.iloc[i]['location_x'], pass_df1_avg_loc.iloc[i]['location_y'], 
                             color='#90EE90', edgecolors='black', s=600, linewidths=2)
                ax11.text(pass_df1_avg_loc.iloc[i]['location_x'], pass_df1_avg_loc.iloc[i]['location_y'], 
                          s=pass_df1_avg_loc.iloc[i]['Jersey No.'], color='black', weight='bold', 
                          ha='center', va='center', fontsize=15, fontname="Georgia", zorder=2)
            ax11.set_title(f'{team_name1} - {formation1} (0 to {sub_min1} mins)', fontsize=14, fontweight='bold', fontname="Georgia", y=0.97)
            pitch11.kdeplot(x=pass_df1.location_x,y=pass_df1.location_y,cmap="Greens",
                shade=True,
                n_levels=10,
                alpha=0.5,
                zorder=0,ax=ax11 ,
                    linewidths=0 
                
            )
            
            
           
            
            
            
            
            plt.show()
            st.subheader(f"Pass Network")
            st.pyplot(fig11)        
            st.write("No. of successfull passes:", successful_pass1_number)     
            st.write("No. of successfull progressive passes:", successful_progressive_pass1_number)
            st.metric(label="Pass Accuracy", value=f"{successful_pass1_rate}%", delta=None)
            st.subheader(f"Players Pass Accuracy")
            st.dataframe(pass_rate_player1)



            

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Pass Mapping")
            outcome_options0 = ["All", "Successful", "Incomplete", 'Pass Offside', "Out", "Unknown", "Injury Clearance"]
            progressive_options0 = ["Both", "Yes", "No"]
            player_pass0=['All']+pass_df0.player.unique().tolist()
            pass_type_list0=['All']+pass_df0.pass_type.unique().tolist()
            # Selectbox for pass outcome and progressive pass filter
            outcome_filter0 = st.selectbox("Select Pass Outcome", outcome_options0)
            progressive_filter0 = st.selectbox("Is Progressive Pass?", progressive_options0)
            type_pass0=st.selectbox("Select the Pass Type",pass_type_list0)
            player_pass0 = st.selectbox("Select Player", player_pass0)
            
            # Filter pass data based on the selected outcomes and progressive pass filter
            pass_map_df0 = pass_df0

            
            if outcome_filter0 != "All":
                pass_map_df0 = pass_map_df0[pass_map_df0['pass_outcome'] == outcome_filter0]
            if player_pass0!= 'All':
                pass_map_df0 = pass_map_df0[pass_map_df0['player'] == player_pass0]
            if type_pass0!= 'All':
                pass_map_df0 = pass_map_df0[pass_map_df0['pass_type'] == type_pass0]   
            if progressive_filter0 == "Yes":
                pass_map_df0 = pass_map_df0[pass_map_df0['progressive_pass'] == 1]
            elif progressive_filter0 == "No":
                pass_map_df0 = pass_map_df0[pass_map_df0['progressive_pass'] == 0]
        
            # Plotting the pass data
            pitch000 = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
            fig000, ax000 = plt.subplots(figsize=(10, 6))
            pitch000.draw(ax=ax000)
        
            for i in range(pass_map_df0.shape[0]):
                start_x = pass_map_df0.iloc[i]['location_x']
                start_y = pass_map_df0.iloc[i]['location_y']
                end_x = pass_map_df0.iloc[i]['pass_end_location_x']
                end_y = pass_map_df0.iloc[i]['pass_end_location_y']
                outcome = pass_map_df0.iloc[i]['pass_outcome']
        
                # Set color based on pass outcome
                if pd.isna(outcome):         # NaN outcome
                    color = "black"
                elif outcome == "Incomplete":
                    color = "red"
                elif outcome == "Out":
                    color = "orange"
                elif outcome == "Unknown":
                    color = "blue"
                elif outcome == "Injury Clearance":
                    color = "green"
                elif outcome == "Pass Offside":
                    color = "purple" #["All", "Successful", "Incomplete", 'Pass Offside', "Out", "Unknown", "Injury Clearance"]
        
                # Draw arrow at the end
                arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                        mutation_scale=15,  # Controls the size of the arrowhead
                                        color=color, lw=1,zorder=1)
                ax000.add_patch(arrow)
                pitch000.kdeplot(end_x,end_y,cmap="Blues",
                            shade=True,
                            n_levels=10,
                            alpha=0.5,
                            zorder=0,ax=ax000 ,
                                linewidths=0 )
            st.pyplot(fig000)
        
        with col4:
            st.subheader("Pass Mapping")
            outcome_options1 = ["All", "Successful", "Incomplete", 'Pass Offside', "Out", "Unknown", "Injury Clearance"]
            progressive_options1 = ["Both", "Yes", "No"]
            pass_type_list1=['All']+pass_df1.pass_type.unique().tolist()
            player_pass1=['All']+pass_df1.player.unique().tolist()
            # Selectbox for pass outcome and progressive pass filter
            outcome_filter1 = st.selectbox("Select Pass Outcome", outcome_options1, key="outcome_filter1")
            progressive_filter1 = st.selectbox("Is Progressive Pass?", progressive_options1, key="progressive_filter1")
            type_pass1=st.selectbox("Select the Pass Type",pass_type_list1)            
            player_pass1 = st.selectbox("Select Player", player_pass1)
            
            # Filter pass data based on the selected outcomes and progressive pass filter
            # Plotting the pass data
            
            
            pass_map_df1 = pass_df1
            if outcome_filter1 != "All":
                pass_map_df1 = pass_map_df1[pass_map_df1['pass_outcome'] == outcome_filter1]
            if player_pass1!= 'All':
                pass_map_df1 = pass_map_df1[pass_map_df1['player'] == player_pass1]
            if type_pass1!= 'All':
                pass_map_df1 = pass_map_df1[pass_map_df1['pass_type'] == type_pass1] 
            if progressive_filter1 == "Yes":
                pass_map_df1 = pass_map_df1[pass_map_df1['progressive_pass'] == 1]
            elif progressive_filter1 == "No":
                pass_map_df1 = pass_map_df1[pass_map_df1['progressive_pass'] == 0]
        
            pitch111 = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
            fig111, ax111 = plt.subplots(figsize=(10, 6))
            pitch111.draw(ax=ax111)
        
            for i in range(pass_map_df1.shape[0]):
                start_x1 = pass_map_df1.iloc[i]['location_x']
                start_y1 = pass_map_df1.iloc[i]['location_y']
                end_x1 = pass_map_df1.iloc[i]['pass_end_location_x']
                end_y1 = pass_map_df1.iloc[i]['pass_end_location_y']
                outcome1 = pass_map_df1.iloc[i]['pass_outcome']
        
                # Set color based on pass outcome
                if pd.isna(outcome1):         # NaN outcome
                    color = "black"
                elif outcome1 == "Incomplete":
                    color = "red"
                elif outcome1 == "Out":
                    color = "orange"
                elif outcome1 == "Unknown":
                    color = "blue"
                elif outcome1 == "Injury Clearance":
                    color = "green"
                elif outcome1 == "Pass Offside":
                    color = "purple"
                # Draw arrow at the end
                arrow1 = FancyArrowPatch((start_x1, start_y1), (end_x1, end_y1),
                                        mutation_scale=15,  # Controls the size of the arrowhead
                                        color=color, lw=1,zorder=1)
                ax111.add_patch(arrow1)
                pitch111.kdeplot(end_x1,end_y1,cmap="Greens",
                shade=True,
                n_levels=10,
                alpha=0.5,
                zorder=0,ax=ax111 ,
                    linewidths=0 )   
            st.pyplot(fig111)
