from torch.utils.data import Dataset
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

class PlaysData(Dataset):
    variants_output_size = {1:5,2:1,3:3,4:5}

    def __init__(self, variant, data=None, all=False):
        self.v = variant
        self.all = all
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
        self.data["amount_of_players_causing_pressure_on_qb_during_snap"] = []

        self.data["result"] = []
        
    def process_plays(self):
        #iteration over all tracking... csvs
        week_df_i = 0
        for week_df in tqdm(self.tracking):
            week_df = week_df.merge(self.players[['nflId', 'position']], on='nflId', how='left')
            merged = week_df.merge(self.plays, on=['gameId', 'playId'], how='inner')

            #iteration over all plays 
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
                
                if self.all: 
                    amount_of_qb_frames = len(qb_data)
                else:
                    amount_of_qb_frames = 1
                    
                # iteration over play qb frames
                for i in range(amount_of_qb_frames):
                    if not self.all:
                        i = -1

                    qb_snap = qb_data.sort_values('frameId').iloc[i]
                    ball_frame = qb_snap['frameId']

                    # getting qb features
                    self.data["qb_x"].append(qb_snap['x'])
                    self.data["qb_y"].append(qb_snap['y'])
                    self.data["qb_orientation"].append(qb_snap['o'])
                    self.data["qb_speed"].append(qb_snap['s'])
                    self.data["qb_direction"].append(qb_snap['dir'])
                    self.data["qb_accel"].append(qb_snap['a'])
                    
                    # getting receivers features
                    targetedReceiver = None
                    self.receivers = play_players[play_players['routeRan'].notna()].copy()

                    self.receivers = self.receivers.merge(self.players[['nflId', 'position']], on='nflId', how='left')
                    self.sorting_receivers(play_df, ball_frame)

                    for j in range(5):
                        if j < len(self.receivers):
                            r = self.receivers.iloc[j]
                            rid = r['nflId']
                            r_data = play_df[(play_df['nflId'] == rid) & (play_df["frameId"] == ball_frame)]
                            if not r_data.empty:
                                r_snap = r_data.iloc[0]
                                self.data[f"x_{j}"].append(r_snap['x'])
                                self.data[f"y_{j}"].append(r_snap['y'])
                                self.data[f"vel_{j}"].append(r_snap['s'])
                                self.data[f"accel_{j}"].append(r_snap['a'])
                                self.data[f"orientation_{j}"].append(r_snap['o'])
                                dist = ((r_snap['x'] - qb_snap['x']) ** 2 + (r_snap['y'] - qb_snap['y']) ** 2) ** 0.5
                                self.data[f"dist_qb_{j}"].append(dist)
                                self.data[f"receiver_type_{j}"].append(r['position'])
                                if r["wasTargettedReceiver"]:
                                    targetedReceiver = j

                                # getting defenders features
                                defenders = play_df[play_df['position'].isin(['CB', 'S', 'LB', 'FS', 'SS', 'DE', 'DT'])].copy()
                                defenders['dist'] = ((defenders['x'] - r_snap['x']) ** 2 +
                                                    (defenders['y'] - r_snap['y']) ** 2) ** 0.5
                                closest = defenders.nsmallest(2, 'dist')

                                for k in range(2):
                                    if k < len(closest):
                                        d = closest.iloc[k]
                                        self.data[f"defensor_x_{j}_{k}"].append(d['x'])
                                        self.data[f"defensor_y_{j}_{k}"].append(d['y'])
                                        self.data[f"defensor_vel_{j}_{k}"].append(d['s'])
                                        self.data[f"defensor_accel_{j}_{k}"].append(d['a'])
                                        self.data[f"defensor_orientation_{j}_{k}"].append(d['o'])
                                    else:
                                        for field in ['x', 'y', 'vel', 'accel', 'orientation']:
                                            self.data[f"defensor_{field}_{j}_{k}"].append(None)
                            else:
                                for field in ['x', 'y', 'vel', 'accel', 'orientation', 'dist_qb', 'receiver_type']:
                                    self.data[f"{field}_{k}"].append(None)
                                for j in range(2):
                                    for field in ['x', 'y', 'vel', 'accel', 'orientation']:
                                        self.data[f"defensor_{field}_{j}_{k}"].append(None)
                        else:
                            for field in ['x', 'y', 'vel', 'accel', 'orientation', 'dist_qb', 'receiver_type']:
                                self.data[f"{field}_{k}"].append(None)
                            for j in range(2):
                                for field in ['x', 'y', 'vel', 'accel', 'orientation']:
                                    self.data[f"defensor_{field}_{j}_{k}"].append(None)

                    amount_causing_pressure = 0

                    for player in play_players.itertuples():
                        if player.causedPressure:
                            amount_causing_pressure += 1

                    self.data["amount_of_players_causing_pressure_on_qb_during_snap"].append(amount_causing_pressure)

                    if self.v == 1 or self.v == 4:
                        self.data["result"].append(targetedReceiver)
                    elif self.v == 2:
                        play_row = play_df[(play_df["gameId"] == gameId) & (play_df['playId'] == playId)]
                        self.data["result"].append(play_row['dis'].sum())
                    elif self.v == 3 or self.v == 5 or self.v == 6:
                        pass_result = play_info.iloc[0]['passResult']
                        if self.v == 5 or self.v == 3:
                            if pass_result != "R" and pass_result != "S":
                                self.data["result"].append(pass_result)
                            else:
                                self.data["result"].append(None)
                        elif self.v == 6:
                            if pass_result != "R" and pass_result != "S" and pass_result != "IN":
                                self.data["result"].append(pass_result)
                            else:
                                self.data["result"].append(None)


        week_df_i += 1        
    def sorting_receivers(self, play_df, ball_frame):
        frame = play_df[play_df['frameId'] == ball_frame]
        receiver_positions = frame[frame['nflId'].isin(self.receivers['nflId'])][['nflId', 'y']]

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
            name = f"./finalFeatures/final_data_variant{self.v}.csv"
        self.data.to_csv(name, index=False)




