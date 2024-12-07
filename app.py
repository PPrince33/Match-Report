import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import streamlit as st
from statsbombpy import sb
from mplsoccer import Pitch
import streamlit as st
from matplotlib.patches import Polygon
import io
from PIL import Image

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
        ax.text(poss0 / 2, 0, f"{team_name0} - {poss0 * 100:.1f}%",fontsize=14, ha='center', va='center', fontweight='bold', color='black',fontname="Monospace")
        ax.text(poss0 + poss1 / 2, 0, f"{team_name1} - {poss1 * 100:.1f}%",fontsize=14, ha='center', va='center', fontweight='bold', color='black',fontname="Monospace")
        plt.suptitle("Possession", fontsize=14, fontweight='bold',fontname="Monospace",y=1.1)
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

        
        



        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{team_name0} XI")
            st.dataframe(startingXI0, height=400, width=700)
            st.markdown(f"Substitutions")
            st.dataframe(sub0, height=150, width=700)
        with col2:
            st.subheader(f"{team_name1} XI")
            st.dataframe(startingXI1, height=400, width=700)
            st.markdown(f"Substitutions")
            st.dataframe(sub1, height=150, width=700)
        
        st.markdown("<h2 style='text-align: center;'>Average Formation</h2>", unsafe_allow_html=True)
        min_passes_options = [1, 2, 3, 4, 5,6 ,7, 8, 9, 10, 11, 12, 13]
        
        
        col1, col2 = st.columns(2)
        
        # Display the DataFrames in the columns
        with col1:
            
            pitch0 = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
            fig0, ax0 = plt.subplots(figsize=(10, 6))
            pitch0.draw(ax=ax0)
            for i in range(all_avg_loc0.shape[0]):
                pitch0.scatter(all_avg_loc0['location_x'][i], all_avg_loc0['location_y'][i], 
                               color='#DEEFF5', edgecolors='black', s=600, ax=ax0)
                pitch0.text(all_avg_loc0['location_x'][i], all_avg_loc0['location_y'][i], 
                            s=all_avg_loc0['jersey_number'][i], color='black', weight='bold', 
                            ha='center', va='center', fontsize=15, fontname="Monospace", zorder=2, ax=ax0)
            ax0.set_title(f'{team_name0} - {formation0}', fontsize=14, fontweight='bold', fontname="Monospace", y=0.97)
            plt.show()
            
            st.pyplot(fig0)

        with col2:
                           
            pitch1 = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            pitch1.draw(ax=ax1)
            for i in range(all_avg_loc1.shape[0]):
                pitch1.scatter(all_avg_loc1['location_x'][i], all_avg_loc1['location_y'][i], 
                               color='#90EE90', edgecolors='black', s=600, ax=ax1)
                pitch1.text(all_avg_loc1['location_x'][i], all_avg_loc1['location_y'][i], 
                            s=all_avg_loc1['jersey_number'][i], color='black', weight='bold', 
                            ha='center', va='center', fontsize=15, fontname="Monospace", zorder=2, ax=ax1)
            ax1.set_title(f'{team_name1} - {formation1}', fontsize=14, fontweight='bold', fontname="Monospace", y=0.97)
            plt.show()       
            
            st.pyplot(fig1) 

        st.markdown("<h2 style='text-align: center;'>Pass Analysis</h2>", unsafe_allow_html=True)
        min_passes = st.selectbox("Select Minimum Passes (Pass Network)", min_passes_options)
        col1, col2 = st.columns(2)
        
        with col1:

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
                          ha='center', va='center', fontsize=15, fontname="Monospace", zorder=2)
            ax00.set_title(f'{team_name0} - {formation0} (0 to {sub_min0} mins)', fontsize=14, fontweight='bold', fontname="Monospace", y=0.97)
            pitch00.kdeplot(x=pass_df0.location_x,y=pass_df0.location_y,cmap="Blues",
                shade=True,
                n_levels=10,
                alpha=0.5,
                zorder=0,ax=ax00 ,
                    linewidths=0)
            plt.show()
            
            st.pyplot(fig00)
            st.write("No. of successfull passes:", successful_pass0_number)        
            st.write("No. of successfull progressive passes:", successful_progressive_pass0_number)
            st.metric(label="Pass Accuracy", value=f"{successful_pass0_rate}%", delta=None)
            st.markdown(f"Players Pass Accuracy")
            st.dataframe(pass_rate_player0)   
            

        with col2:             
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
                          ha='center', va='center', fontsize=15, fontname="Monospace", zorder=2)
            ax11.set_title(f'{team_name1} - {formation1} (0 to {sub_min1} mins)', fontsize=14, fontweight='bold', fontname="Monospace", y=0.97)
            pitch11.kdeplot(x=pass_df1.location_x,y=pass_df1.location_y,cmap="Greens",
                shade=True,
                n_levels=10,
                alpha=0.5,
                zorder=0,ax=ax11 ,
                    linewidths=0 
                
            )
            
            
           
            
            
            
            
            plt.show()
            
            st.pyplot(fig11)        
            st.write("No. of successfull passes:", successful_pass1_number)     
            st.write("No. of successfull progressive passes:", successful_progressive_pass1_number)
            st.metric(label="Pass Accuracy", value=f"{successful_pass1_rate}%", delta=None)
            st.markdown(f"Players Pass Accuracy")
            st.dataframe(pass_rate_player1)



            


        
        # Define colors for pass outcomes
        outcome_colors = {
            "Incomplete": "red",
            "Out": "orange",
            "Unknown": "#DEEFF5",
            "Injury Clearance": "green",
            "Pass Offside": "purple",
            "Successful": "black"
        }
        
        def plot_pass_map(pass_df, outcome_filter, progressive_filter, player_filter, pass_type_filter, kde_color,team_name):
            # Filter data based on user input
            filtered_df = pass_df
            pass_df.pass_outcome.fillna('Successful',inplace=True)
            if outcome_filter != "All":
                filtered_df = filtered_df[filtered_df['pass_outcome'] == outcome_filter]
            if player_filter != "All":
                filtered_df = filtered_df[filtered_df['player'] == player_filter]
            if pass_type_filter != "All":
                filtered_df = filtered_df[filtered_df['pass_type'] == pass_type_filter]
            if progressive_filter == "Yes":
                filtered_df = filtered_df[filtered_df['progressive_pass'] == 1]
            elif progressive_filter == "No":
                filtered_df = filtered_df[filtered_df['progressive_pass'] == 0]
        
            # Plotting
            pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
            fig, ax = plt.subplots(figsize=(10, 6))
            pitch.draw(ax=ax)
            
            for _, row in filtered_df.iterrows():
                start_x, start_y = row['location_x'], row['location_y']
                end_x, end_y = row['pass_end_location_x'], row['pass_end_location_y']
                outcome = row['pass_outcome']
                color = outcome_colors.get(outcome, "black")  # default to black for NaN or 'Successful'
                
                # Draw pass arrow
                arrow = FancyArrowPatch((start_x, start_y), (end_x, end_y), mutation_scale=15, color=color, lw=1, zorder=1)
                ax.add_patch(arrow)
        
            # KDE Plot
            pitch.kdeplot(
                filtered_df['pass_end_location_x'], filtered_df['pass_end_location_y'], 
                cmap=kde_color, shade=True, n_levels=10, alpha=0.5, zorder=0, ax=ax, linewidths=0
            )
            ax.set_title(f'{team_name}:{player_filter}', fontsize=14, fontweight='bold', fontname="Monospace", y=0.97)
            st.write("No. of passes after filter:", filtered_df.shape[0]) 
            st.write("No. of successful passes after filter:", filtered_df[filtered_df.pass_outcome=='Successful'].shape[0]) 
            return fig
        
        # Streamlit layout
        st.markdown("<h2 style='text-align: center;'>Pass Mapping</h2>", unsafe_allow_html=True)

        col3, col4 = st.columns(2)
        
        # Column 3 for pass_df0
        with col3:
            
            outcome_options0 = ["All", "Successful", "Incomplete", 'Pass Offside', "Out", "Unknown", "Injury Clearance"]
            progressive_options0 = ["Both", "Yes", "No"]
            player_pass0 = ['All'] + pass_df0['player'].unique().tolist()
            pass_type_list0 = ['All'] + pass_df0['pass_type'].unique().tolist()
            
            outcome_filter0 = st.selectbox("Select Pass Outcome", outcome_options0)
            progressive_filter0 = st.selectbox("Is Progressive Pass?", progressive_options0)
            player_pass0 = st.selectbox("Select Player", player_pass0)
            type_pass0 = st.selectbox("Select Pass Type", pass_type_list0)
            
            fig0 = plot_pass_map(pass_df0, outcome_filter0, progressive_filter0, player_pass0, type_pass0, kde_color="Blues",team_name=team_name0)
            st.pyplot(fig0)
            
        # Column 4 for pass_df1
        with col4:
        
            outcome_options1 = ["All", "Successful", "Incomplete", 'Pass Offside', "Out", "Unknown", "Injury Clearance"]
            progressive_options1 = ["Both", "Yes", "No"]
            player_pass1 = ['All'] + pass_df1['player'].unique().tolist()
            pass_type_list1 = ['All'] + pass_df1['pass_type'].unique().tolist()
            
            outcome_filter1 = st.selectbox("Select Pass Outcome", outcome_options1, key="outcome_filter1")
            progressive_filter1 = st.selectbox("Is Progressive Pass?", progressive_options1, key="progressive_filter1")
            player_pass1 = st.selectbox("Select Player", player_pass1, key="player_pass1")
            type_pass1 = st.selectbox("Select Pass Type", pass_type_list1, key="type_pass1")
            
            fig1 = plot_pass_map(pass_df1, outcome_filter1, progressive_filter1, player_pass1, type_pass1, kde_color="Greens",team_name=team_name1)
            st.pyplot(fig1)

        

#-----------------------------------------------------------------------------------------------------------------------------------------

        # Title for the app
        st.markdown("<h2 style='text-align: center;'>Shot Analysis</h2>", unsafe_allow_html=True)
        
        # Initialize final_shot_df
        final_shot_df = pd.DataFrame()
        shot_df = event_df[event_df.type == 'Shot']
        shot_df = shot_df.dropna(how='all', axis=1).reset_index(drop=True)
        
        for i in range(len(shot_df)):
            shot_surrounding_data = shot_df.at[i, 'shot_freeze_frame']
            if shot_surrounding_data and isinstance(shot_surrounding_data, list):  # Ensure it's a non-empty list
                formatted_shot_surrounding_data = [
                    {
                        "player_location": item["location"],
                        "player_name": item["player"]["name"],
                        "position_name": item["position"]["name"],
                        "teammate": item["teammate"]
                    }
                    for item in shot_surrounding_data
                ]
                formatted_shot_surrounding_df = pd.DataFrame(formatted_shot_surrounding_data)
        
                full_detail_shot_df = pd.concat(
                    [formatted_shot_surrounding_df, pd.concat([shot_df.loc[[i]]] * len(formatted_shot_surrounding_df), ignore_index=True)],
                    axis=1
                )
                final_shot_df = pd.concat([final_shot_df, full_detail_shot_df], axis=0, ignore_index=True)
        
        # Add penalty shots directly
        final_shot_df = pd.concat([final_shot_df, shot_df[shot_df.shot_type == 'Penalty']], axis=0, ignore_index=True)
        
        # Extract player location x and y
        final_shot_df['player_location_x'] = final_shot_df['player_location'].apply(
            lambda loc: loc[0] if isinstance(loc, (list, tuple)) and len(loc) > 0 else None
        )
        final_shot_df['player_location_y'] = final_shot_df['player_location'].apply(
            lambda loc: loc[1] if isinstance(loc, (list, tuple)) and len(loc) > 1 else None
        )
        
        # Create unique shot ID
        final_shot_df['shot_id'] = (
            final_shot_df['player'].astype(str) + ' (' +
            final_shot_df['minute'].astype(str) + ':' +
            final_shot_df['second'].astype(str) + ')'
        )
        
        # Merge jersey number information
        full_jersey_df = pd.concat([full_lineup_expanded0, full_lineup_expanded1])
        full_jersey_df = full_jersey_df[['player_name', 'jersey_number']]
        final_shot_df = pd.merge(final_shot_df, full_jersey_df, on='player_name', how='left')
        
        # Handle missing jersey numbers and convert to integers
        final_shot_df['jersey_number'] = final_shot_df['jersey_number'].fillna(0).astype(int)
        
        # Split data by team
        final_shot_df0 = final_shot_df[final_shot_df.team == team_name0]
        final_shot_df1 = final_shot_df[final_shot_df.team == team_name1]
        
        # Add player team information
        final_shot_df0.loc[:, 'player_team'] = np.where(
            final_shot_df0['teammate'] == False, team_name1, final_shot_df0['team']
        )
        final_shot_df1.loc[:, 'player_team'] = np.where(
            final_shot_df1['teammate'] == False, team_name0, final_shot_df1['team']
        )
        
        # Unique shot IDs
        selected_shot0 = final_shot_df0['shot_id'].unique().tolist()
        selected_shot1 = final_shot_df1['shot_id'].unique().tolist()
        
        # Streamlit columns
        col1, col2 = st.columns(2)
        
        # Function to plot shot mapping
        def plot_shot_mapping(shot_mapping, team_name0, team_name1, rotate=False):
            # Define team colors
            team_name0_color = '#DEEFF5'  # Light blue
            team_name1_color = '#90EE90'  # Light green
            
            # Assign colors based on the team
            shot_mapping['color'] = shot_mapping['player_team'].apply(
                lambda team: team_name0_color if team == team_name0 else team_name1_color
            )
            
            # Create a pitch
            pitch = Pitch(pitch_type='statsbomb', pitch_color='white', line_color='black')
            fig, ax = plt.subplots(figsize=(10, 7))  # Adjust figure size as needed
            pitch.draw(ax=ax)
            if shot_mapping.shape[0]==1:
                ax.scatter(
                            shot_mapping['location_x'],  # X-coordinate
                            shot_mapping['location_y'],  # Y-coordinate
                            color='red',  # Color for specific players
                            edgecolors='black', 
                            zorder=3, 
                            s=1000 * (shot_mapping['shot_statsbomb_xg'].iloc[0] if pd.notnull(shot_mapping['shot_statsbomb_xg'].iloc[0]) else 50)
                        )
                                        # Add triangles and lines
                triangle_vertices = [
                    (shot_mapping['location_x'][0], shot_mapping['location_y'][0]),
                    (120, 36),
                    (120, 44)
                ]
                
                triangle = Polygon(
                        triangle_vertices, closed=True, color='lightcoral', edgecolor='black', alpha=0.2, zorder=1
                    )
                ax.add_patch(triangle)
                ax.plot(
                    [shot_mapping['location_x'], shot_mapping['shot_end_location_x']],  # X-coordinates
                    [shot_mapping['location_y'], shot_mapping['shot_end_location_y']],  # Y-coordinates
                    color='blue', linewidth=1, zorder=2, linestyle='--'
                )
                ax.set_xlim(95, 121)
                ax.axis('off')  # Turn off the axes for a cleaner look
                if rotate:
                    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Ensure no padding
                    fig.patch.set_alpha(0)  # Transparent background for cleaner rotation
                
                return fig
            else:
            # Plot scatter points
                for i in range(shot_mapping.shape[0]):
                    ax.scatter(
                        shot_mapping['player_location_x'].iloc[i],  # X-coordinate
                        shot_mapping['player_location_y'].iloc[i],  # Y-coordinate
                        color=shot_mapping['color'].iloc[i],  # Color based on the team
                        edgecolors='black', zorder=3,  # Black border
                        s=400  # Size of the marker
                    )
                    ax.scatter(
                        shot_mapping['location_x'].iloc[i],  # X-coordinate
                        shot_mapping['location_y'].iloc[i],  # Y-coordinate
                        color='red',  # Color for specific players
                        edgecolors='black', zorder=3, s=1000 * (shot_mapping['shot_statsbomb_xg'].iloc[i] if pd.notnull(shot_mapping['shot_statsbomb_xg'].iloc[i]) else 50)
                    )
                    ax.text(shot_mapping.iloc[i]['player_location_x'], shot_mapping.iloc[i]['player_location_y'], 
                          s=shot_mapping.iloc[i]['jersey_number'], color='black', weight='bold', 
                          ha='center', va='center', fontsize=10, fontname="Monospace", zorder=4)
                    # Add triangles and lines
                    triangle_vertices = [
                        (shot_mapping['location_x'].iloc[1], shot_mapping['location_y'].iloc[1]),
                        (120, 36),
                        (120, 44)
                    ]
                    triangle = Polygon(
                        triangle_vertices, closed=True, color='lightcoral', edgecolor='black', alpha=0.2, zorder=1
                    )
                    ax.add_patch(triangle)
                    ax.plot(
                        [shot_mapping['location_x'].iloc[i], shot_mapping['shot_end_location_x'].iloc[i]],  # X-coordinates
                        [shot_mapping['location_y'].iloc[i], shot_mapping['shot_end_location_y'].iloc[i]],  # Y-coordinates
                        color='blue', linewidth=1, zorder=2, linestyle='--'
                    )
            
                # Set limits
                ax.set_xlim(min(shot_mapping['player_location_x']) - 10, 121)
                ax.axis('off')  # Turn off the axes for a cleaner look
                
                # Apply rotation if specified
                if rotate:
                    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Ensure no padding
                    fig.patch.set_alpha(0)  # Transparent background for cleaner rotation
                
                return fig
        col1, col2 = st.columns(2)
        # Column 1 (Rotated Image)
        with col1:
            selected_shot = st.selectbox("Select a Shot", options=selected_shot0)
            shot_mapping0 = final_shot_df0[final_shot_df0['shot_id'] == selected_shot].reset_index(drop=True)
            fig0 = plot_shot_mapping(shot_mapping0, team_name0, team_name1, rotate=True)
            
            # Save the figure to a buffer
            buf = io.BytesIO()
            fig0.savefig(buf, format="jpg", facecolor='white')
            buf.seek(0)
            
            # Convert to image and rotate 90 degrees counterclockwise
            image0 = Image.open(buf)
            image0 = image0.rotate(90, expand=True)
            
            # Crop the top part of the image (adjust the crop box as needed)
            width, height = image0.size
            top_crop = height // 4  # Crop 1/5 of the image from the top
            bottom_crop = height // 4  # Crop 1/5 of the image from the bottom
            crop_box = (0, top_crop, width, height - bottom_crop) 
            # Crop the top 1/5 of the image # Crop the top 1/5 of the image
            cropped_image0 = image0.crop(crop_box)
            
            xg_value0 = np.round(shot_mapping0['shot_statsbomb_xg'][0], 2)

            st.image(
                cropped_image0, 
                caption=f"Shot Visualization {team_name0} (xG: {xg_value0},Shot Outcome: {shot_mapping0['shot_outcome'][0]})",
                use_column_width=True
            )
        # Column 2 (Rotated and Cropped Image)
        with col2:
            selected_shot = st.selectbox("Select a Shot", options=selected_shot1)
            shot_mapping1 = final_shot_df1[final_shot_df1['shot_id'] == selected_shot].reset_index(drop=True)
            fig1 = plot_shot_mapping(shot_mapping1, team_name0, team_name1, rotate=True)
            
            # Save the figure to a buffer
            buf = io.BytesIO()
            fig1.savefig(buf, format="jpg", facecolor='white')
            buf.seek(0)
            
            # Convert to image and rotate 90 degrees counterclockwise
            image1 = Image.open(buf)
            image1 = image1.rotate(90, expand=True)
            
            # Crop the top part of the image (adjust the crop box as needed)
            width, height = image1.size
            top_crop = height // 4  # Crop 1/5 of the image from the top
            bottom_crop = height // 4  # Crop 1/5 of the image from the bottom
            crop_box = (0, top_crop, width, height - bottom_crop) 
            # Crop the top 1/5 of the image
            cropped_image1 = image1.crop(crop_box)
            
            xg_value1 = np.round(shot_mapping1['shot_statsbomb_xg'][0], 2)

            st.image(
                cropped_image1, 
                caption=f"Shot Visualization {team_name1} (xG: {xg_value1},Shot Outcome: {shot_mapping1['shot_outcome'][0]})",
                use_column_width=True
            )
