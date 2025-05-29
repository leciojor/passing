from torch.utils.data import Dataset
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class PlaysData(Dataset):
    VARIANTS_OUTPUT_SIZE = {1:5,2:1,3:3,4:5,5:1,6:3}
    # QB will be removed when exception scenarios are removed/considered
    RECEIVER_TYPES = ["WR", "TE", "QB", "RB", "FB"]

    def plot_distributions(data, col, v):
        plt.figure(figsize=(6, 4))
        data[col].value_counts(dropna=False).sort_index().plot(kind='bar')
        plt.title(f'Distribution of {col} variant {v}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"./distributions/distr_{col}_{v}.png")
        plt.clf()


    def __init__(self, variant, data=None, all=False, p=3, i=4, game_id=None, play_id=None, passed_result_extra=False):
        self.i = i
        self.v = variant
        self.passed_extra = passed_result_extra
        self.p = p
        self.all = all
        self.saved = False
        self.amounts_receivers = {}
        if data is None:
            self.players = pd.read_csv("data/players.csv")
            self.player_play = pd.read_csv("data/player_play.csv")
            self.plays = pd.read_csv("data/plays.csv")
            self.tracking = []
            for i in range(1, 10):
                self.tracking.append(pd.read_csv(f"data/tracking_week_{i}.csv"))
            
            self.data = {}

            self.initializing_df_data()
            if not game_id and not play_id:
                self.process_plays()
            else:
                self.game_id = game_id
                self.play_id = play_id
                self.all = True
                self.process_frames_of_play()

            self.data = pd.DataFrame.from_dict(self.data)
        else:
            self.saved = True
            self.data = data
        
        self.length = len(self.data)
        self.col_size = self.data.shape[1]
        

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        row = self.data.iloc[i]
        return torch.tensor(row.iloc[:self.col_size-PlaysData.VARIANTS_OUTPUT_SIZE[self.v]].values, dtype=torch.float32), torch.tensor(row.iloc[self.col_size-PlaysData.VARIANTS_OUTPUT_SIZE[self.v]:].values, dtype=torch.float32)

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

        # pressure fields
        self.data["amount_of_players_causing_pressure_on_qb"] = []
        self.data["yardsToGo"] = []
        self.data["down"] = []
        self.data["yardLine"] = []

        self.data["result"] = []
        if self.passed_extra:
            self.data["passResultExtra"] = []
    
    def process_frames_of_play(self):

        for week_df in tqdm(self.tracking):
            filtered_df = week_df[(week_df["playId"] == self.play_id) & (week_df["gameId"] == self.game_id)]

            if not filtered_df.empty:
                filtered_df = filtered_df.merge(self.players[['nflId', 'position']], on='nflId', how='left')
                play_df = filtered_df.merge(self.plays, on=['gameId', 'playId'], how='inner')

                self.play_df_formation(self.game_id, self.play_id, play_df)

    def play_df_formation(self, gameId, playId, play_df):
        play_players = self.player_play[
            (self.player_play['gameId'] == gameId) &
            (self.player_play['playId'] == playId)
        ]
        play_info = self.plays[
            (self.plays['gameId'] == gameId) &
            (self.plays['playId'] == playId)
        ]

        if not self.all:
            qb_data = play_df[(play_df['position'] == 'QB') & (play_df["event"] == "pass_forward")]
        else:
            qb_data = play_df[(play_df['position'] == 'QB')]

        qb_data = qb_data.sort_values('frameId')
        self.qb_data = qb_data
        if qb_data.empty:
            return False

        if self.all: 
            qb_frames_start, qb_frames_end = self.get_frames_indexes()
        else:
            qb_frames_start = 0
            qb_frames_end = 1
        
        # iteration over play qb frames
        for i in range(qb_frames_start, qb_frames_end):
            if not self.all:
                i = -1

            qb_snap = qb_data.sort_values('frameId').iloc[i]
            ball_frame = qb_snap['frameId']

            # getting qb features
            self.data["qb_x"].append(qb_snap['x'])
            self.data["qb_y"].append(qb_snap['y'])
            # actual shoulder orientation
            self.data["qb_orientation"].append(qb_snap['o'])
            self.data["qb_speed"].append(qb_snap['s'])
            self.data["qb_direction"].append(qb_snap['dir'])
            self.data["qb_accel"].append(qb_snap['a'])
            
            # getting receivers features
            targetedReceiver = None
            self.receivers = play_players[play_players['routeRan'].notna()].copy()

            self.receivers = self.receivers.merge(self.players[['nflId', 'position']], on='nflId', how='left')
            self.sorting_receivers(play_df, ball_frame)
            amount_receivers = len(self.receivers)
            if amount_receivers in self.amounts_receivers:
                self.amounts_receivers[amount_receivers] +=1
            else:
                self.amounts_receivers[amount_receivers] = 1


            for j in range(5):
                if j < amount_receivers:
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
                            self.data[f"{field}_{j}"].append(None)
                        for k in range(2):
                            for field in ['x', 'y', 'vel', 'accel', 'orientation']:
                                self.data[f"defensor_{field}_{j}_{k}"].append(None)
                else:
                    for field in ['x', 'y', 'vel', 'accel', 'orientation', 'dist_qb', 'receiver_type']:
                        self.data[f"{field}_{j}"].append(None)
                    for k in range(2):
                        for field in ['x', 'y', 'vel', 'accel', 'orientation']:
                            self.data[f"defensor_{field}_{j}_{k}"].append(None)

            amount_causing_pressure = 0

            for player in play_players.itertuples():
                if player.causedPressure:
                    amount_causing_pressure += 1

            self.data["amount_of_players_causing_pressure_on_qb"].append(amount_causing_pressure)
            self.data["yardsToGo"].append(play_df.iloc[0]["yardsToGo"].item())
            self.data["down"].append(play_df.iloc[0]["down"].item())
            self.data["yardlineNumber"].append(play_df.iloc[0]["yardlineNumber"].item())

            if self.passed_extra:
                self.data["passResultExtra"].append(play_info.iloc[0]['passResult'].item())
            if self.v == 1 or self.v == 4:
                self.data["result"].append(targetedReceiver)
            elif self.v == 2:
                self.data["result"].append(play_df.iloc[0]['yardsGained'].item())
            elif self.v == 3 or self.v == 5 or self.v == 6:
                pass_result = play_info.iloc[0]['passResult'].item()
                if self.v == 5 or self.v == 3:
                    if pass_result == "C":
                        self.data["result"].append("C")
                    else:
                        self.data["result"].append("I")
                elif self.v == 6:
                    if pass_result == "C" or pass_result == "IN":
                        self.data["result"].append(pass_result)
                    else:
                        self.data["result"].append("I")
            
        return True

    def get_frames_indexes(self):
        self.qb_data.reset_index(inplace=True)
        qb_frames_start = self.qb_data[self.qb_data["event"] == "ball_snap"].index[0] + 1
        qb_frames_end = self.qb_data[self.qb_data["event"] == "pass_forward"].index[0] + 1

        return qb_frames_start, qb_frames_end

    def process_plays(self):
        #iteration over all tracking... csvs
        self.week_df_i = 0
        play_i = 0 
        for week_df in tqdm(self.tracking):
            week_df = week_df.merge(self.players[['nflId', 'position']], on='nflId', how='left')
            merged = week_df.merge(self.plays, on=['gameId', 'playId'], how='inner')

            #iteration over all plays 
            for (gameId, playId), play_df in merged.groupby(['gameId', 'playId']):
                self.gameId = gameId
                self.playId = playId
                
                if self.all and play_i != self.i:
                    play_i += 1
                    continue
                
                if not self.play_df_formation(gameId, playId, play_df):
                    continue
                
                if self.all:
                    break

            if self.all:
                break

        self.week_df_i += 1  

    def sorting_receivers(self, play_df, ball_frame):
        frame = play_df[play_df['frameId'] == ball_frame]
        receiver_positions = frame[frame['nflId'].isin(self.receivers['nflId'])][['nflId', 'y']]

        self.receivers = self.receivers.merge(receiver_positions, on='nflId', how='left')

        self.receivers = self.receivers.sort_values(by='y', ascending=True).head(5)
        
    def converting_numerical_and_cleaning(self, r=False):
        result_parts = []

        #removing initial nans (just based on results or x_0)
        self.data.dropna(subset=['result', 'x_0'], inplace=True)
        
        #adding one hot encoding for discrete features
        for col in tqdm(self.data.columns):
            if pd.api.types.is_numeric_dtype(self.data[col]) and (col != "result" or self.v == 2):
                result_parts.append(self.data[[col]].astype(float))
            elif self.v == 5 and col == "result":
                binary_results = self.data[col].map({'C': 1, 'I': 0})
                result_parts.append(binary_results.to_frame())
            else:
                if any(col == f'receiver_type_{i}' for i in range(5)):
                    self.data[col] = pd.Categorical(self.data[col], categories=PlaysData.RECEIVER_TYPES)

                dummies = pd.get_dummies(self.data[col], prefix=col, dtype=int)
                result_parts.append(dummies)

        self.data = pd.concat(result_parts, axis=1)

        #filling the rest of nans with the average
        self.data.fillna(self.data.mean(), inplace=True)

        if r:
            def round_(x):
                if isinstance(x, (int, float, np.number)) and x != 0:
                    return round(x, 3)
                return x 
            self.data = self.data.applymap(round_)
                
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

    def distributions_analysis(self):
        for col in self.data.columns:
            if not pd.api.types.is_numeric_dtype(self.data[col]) or pd.api.types.is_integer_dtype(self.data[col]) or col == "result":
                PlaysData.plot_distributions(self.data, col, self.v)
        if not self.saved: 
            plt.figure(figsize=(6, 4))
            plt.bar(list(self.amounts_receivers), list(self.amounts_receivers.values()))
            plt.title(f'Distribution of Receivers Amounts variant {self.v}')
            plt.xlabel("Receivers Amount")
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"./distributions/distr_receivers_amount_{self.v}.png")
            plt.clf()


    def get_orientation_based_on_receiver(self, receiver, frameId):
        x_qb = self.data[[frameId]]["qb_x"]
        y_qb = self.data[[frameId]]["qb_y"]
        x_receiver = self.data[[frameId]][f"x_{receiver}"]
        y_receiver = self.data[[frameId]][f"y_{receiver}"]

        dx = x_receiver - x_qb
        dy = y_receiver - y_qb

        self.data[[frameId]]["qb_orientation"] = (np.degrees(np.arctan2(dy, dx))) % 360

        return torch.tensor(self.data[[frameId]])

    def augmentation(self):
        pass

    def get_csv(self, name=False):
        if not name:
            if self.all:
                name = f"./finalFeatures/final_data_variant{self.v}_{self.all}_instance{self.i}.csv"
            else:
                name = f"./finalFeatures/final_data_variant{self.v}_{self.all}.csv"

        self.data.to_csv(name, index=False)




