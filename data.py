from torch.utils.data import Dataset
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

class PlaysData(Dataset):
    variants_output_size = {1:5,2:1,3:3,4:5}

    def __init__(self, variant, data=None):
        self.v = variant
        if data is None:
            self.players = pd.read_csv("data/players.csv")
            self.player_play = pd.read_csv("data/player_play.csv")
            self.plays = pd.read_csv("data/plays.csv")
            self.tracking = []
            for i in range(1, 10):
                self.tracking.append(pd.read_csv(f"data/tracking_week_{i}.csv"))
            
            self.data = {}

            self.initializing_df_data()
            self.process_plays()

            self.data = pd.DataFrame.from_dict(self.data)
        else:
            self.data = data
        
        self.length = len(self.data)
        self.col_size = self.data.shape[1]
        

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        row = self.data.iloc[i]
        return torch.tensor(row.iloc[:self.col_size-PlaysData.variants_output_size[self.v]].values, dtype=torch.float32), torch.tensor(row.iloc[self.col_size-PlaysData.variants_output_size[self.v]:].values, dtype=torch.float32)

    def __str__(self):
        return self.data

    def initializing_df_data(self):
        for i in range(5):
            self.data[f"x_{i}"] = []
            self.data[f"y_{i}"] = []
            self.data[f"vel_{i}"] = []
            self.data[f"accel_{i}"] = []
            self.data[f"orientation_{i}"] = []
            self.data[f"dist_qb_{i}"] = []
            self.data[f"receiver_type_{i}"] = []
            for j in range(2):
                self.data[f"defensor_x_{i}_{j}"] = []
                self.data[f"defensor_y_{i}_{j}"] = []
                self.data[f"defensor_vel_{i}_{j}"] = []
                self.data[f"defensor_accel_{i}_{j}"] = []
                self.data[f"defensor_orientation_{i}_{j}"] = []
        
        self.data["qb_x"] = []
        self.data["qb_y"] = []
        self.data["qb_orientation"] = []
        self.data["qb_speed"] = []
        self.data["qb_direction"] = []
        self.data["qb_accel"] = []
        for i in range(6):
            self.data[f"qb_pressure_{i}"] = []

        self.data["result"] = []
        
    def process_plays(self):
        for week_df in tqdm(self.tracking):
            week_df = week_df.merge(self.players[['nflId', 'position']], on='nflId', how='left')
            merged = week_df.merge(self.plays, on=['gameId', 'playId'], how='inner')

            for (gameId, playId), play_df in merged.groupby(['gameId', 'playId']):
                play_players = self.player_play[
                    (self.player_play['gameId'] == gameId) &
                    (self.player_play['playId'] == playId)
                ]
                play_info = self.plays[
                    (self.plays['gameId'] == gameId) &
                    (self.plays['playId'] == playId)
                ]

                qb_data = play_df[play_df['position'] == 'QB']
                if qb_data.empty:
                    continue
                qb_snap = qb_data.sort_values('frameId').iloc[0]
                ball_snap_frame = qb_snap['frameId']

                self.data["qb_x"].append(qb_snap['x'])
                self.data["qb_y"].append(qb_snap['y'])
                self.data["qb_orientation"].append(qb_snap['o'])
                self.data["qb_speed"].append(qb_snap['s'])
                self.data["qb_direction"].append(qb_snap['dir'])
                self.data["qb_accel"].append(qb_snap['a'])
                
                targetedReceiver = None
                self.receivers = play_players[play_players['routeRan'].notna()].copy()
                # May need to be optimized
                self.receivers = self.receivers.merge(self.players[['nflId', 'position']], on='nflId', how='left')
                self.sorting_receivers(play_df, ball_snap_frame)

                for i in range(5):
                    if i < len(self.receivers):
                        r = self.receivers.iloc[i]
                        rid = r['nflId']
                        r_data = play_df[play_df['nflId'] == rid].sort_values('frameId')
                        if not r_data.empty:
                            r_snap = r_data.iloc[0]
                            self.data[f"x_{i}"].append(r_snap['x'])
                            self.data[f"y_{i}"].append(r_snap['y'])
                            self.data[f"vel_{i}"].append(r_snap['s'])
                            self.data[f"accel_{i}"].append(r_snap['a'])
                            self.data[f"orientation_{i}"].append(r_snap['o'])
                            dist = ((r_snap['x'] - qb_snap['x']) ** 2 + (r_snap['y'] - qb_snap['y']) ** 2) ** 0.5
                            self.data[f"dist_qb_{i}"].append(dist)
                            self.data[f"receiver_type_{i}"].append(r['position'])
                            if r["wasTargettedReceiver"]:
                                targetedReceiver = i

                            defenders = play_df[play_df['position'].isin(['CB', 'S', 'LB', 'FS', 'SS', 'DE', 'DT'])].copy()
                            defenders['dist'] = ((defenders['x'] - r_snap['x']) ** 2 +
                                                (defenders['y'] - r_snap['y']) ** 2) ** 0.5
                            closest = defenders.nsmallest(2, 'dist')

                            for j in range(2):
                                if j < len(closest):
                                    d = closest.iloc[j]
                                    self.data[f"defensor_x_{i}_{j}"].append(d['x'])
                                    self.data[f"defensor_y_{i}_{j}"].append(d['y'])
                                    self.data[f"defensor_vel_{i}_{j}"].append(d['s'])
                                    self.data[f"defensor_accel_{i}_{j}"].append(d['a'])
                                    self.data[f"defensor_orientation_{i}_{j}"].append(d['o'])
                                else:
                                    for field in ['x', 'y', 'vel', 'accel', 'orientation']:
                                        self.data[f"defensor_{field}_{i}_{j}"].append(None)
                        else:
                            for field in ['x', 'y', 'vel', 'accel', 'orientation', 'dist_qb', 'receiver_type']:
                                self.data[f"{field}_{i}"].append(None)
                            for j in range(2):
                                for field in ['x', 'y', 'vel', 'accel', 'orientation']:
                                    self.data[f"defensor_{field}_{i}_{j}"].append(None)
                    else:
                        for field in ['x', 'y', 'vel', 'accel', 'orientation', 'dist_qb', 'receiver_type']:
                            self.data[f"{field}_{i}"].append(None)
                        for j in range(2):
                            for field in ['x', 'y', 'vel', 'accel', 'orientation']:
                                self.data[f"defensor_{field}_{i}_{j}"].append(None)

                self.data["qb_pressure_3"].append(ball_snap_frame) 
                qb_sack = 1 if not play_info.empty and play_info.iloc[0]['passResult'] == 'Sack' else 0
                self.data["qb_pressure_4"].append(qb_sack)

                time_to_sack = None
                time_to_pressure = None
                caused_pressure = 0
                unblocked = 0 

                near_qb = play_df[
                    (play_df['frameId'] >= ball_snap_frame) &
                    (play_df['frameId'] <= ball_snap_frame + 20) &
                    (play_df['position'].isin(['LB', 'CB', 'S', 'FS', 'SS', 'DE', 'DT']))
                ]

                for frame_id in sorted(near_qb['frameId'].unique()):
                    frame = near_qb[near_qb['frameId'] == frame_id]
                    for _, defender in frame.iterrows():
                        dist = ((defender['x'] - qb_snap['x']) ** 2 + (defender['y'] - qb_snap['y']) ** 2) ** 0.5
                        if dist < 2:
                            if time_to_pressure is None:
                                time_to_pressure = (frame_id - ball_snap_frame) / 10.0
                                caused_pressure = 1

                if qb_sack:
                    last_frame = qb_data['frameId'].max()
                    time_to_sack = (last_frame - ball_snap_frame) / 10.0

                self.data["qb_pressure_0"].append(unblocked)
                self.data["qb_pressure_1"].append(time_to_sack or 0)
                self.data["qb_pressure_2"].append(caused_pressure)
                self.data["qb_pressure_5"].append(time_to_pressure or 0)
                if self.v == 1 or self.v == 4:
                    self.data["result"].append(targetedReceiver)
                elif self.v == 2:
                    play_row = play_df[(play_df["gameId"] == gameId) & (play_df['playId'] == playId)]
                    self.data["result"].append(play_row['dis'].sum())
                elif self.v == 3:
                    self.data["result"].append(play_info.iloc[0]['passResult'])

    def sorting_receivers(self, play_df, ball_snap_frame):
        snap_frame = play_df[play_df['frameId'] == ball_snap_frame]
        receiver_positions = snap_frame[snap_frame['nflId'].isin(self.receivers['nflId'])][['nflId', 'y']]

        self.receivers = self.receivers.merge(receiver_positions, on='nflId', how='left')

        self.receivers = self.receivers.sort_values(by='y', ascending=True).head(5)
        
    def converting_numerical_and_cleaning(self, r=False):
        result = []

        #removing initial nans (just based on result)
        self.data.dropna(subset=['result'], inplace=True)
        
        for col in tqdm(self.data.columns):
            if pd.api.types.is_numeric_dtype(self.data[col]) and (col != "result" or self.v == 2):
                result.append(self.data[col].astype(float))
            else:
                one_hot = pd.get_dummies(self.data[col], prefix=col)
                one_hot = one_hot.astype(int)
                for col_new in one_hot.columns:
                    result.append(one_hot[col_new])

        self.data = pd.concat(result, axis=1)  
        self.data = self.data.astype("float")
        if r:
            def round_(x):
                if isinstance(x, (int, float, np.number)) and x != 0:
                    return round(x, 3)
                return x 
            self.data = self.data.applymap(round_)
        
        #filling the rest of nans with the average
        self.data = self.data.apply(lambda col: col.fillna(col.mean()))
        
        self.data.reset_index(drop=True, inplace=True)
        self.length = len(self.data)

        self.col_size = self.data.shape[1]

        #maybe also normalizing values?
        
    def check_nan_features(self):
        nan_columns = self.data.columns[self.data.isna().any()].tolist()
        for col in nan_columns:
            nan_count = self.data[col].isna().sum()
            print(f"{col}: {nan_count} NaNs")

    def correlation_analysis(self):
        pass
    
    def augmentation(self):
        pass

    def get_csv(self, name=False):
        if not name:
            name = f"./final_data_variant{self.v}.csv"
        self.data.to_csv(name, index=False)




