import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, List, Optional
from matplotlib.patheffects import withStroke
from archs import DeepQBVariant1
import torch
from helpers import getting_frames_dataset

if torch.cuda.is_available():
  DEVICE = torch.device("cuda")
else:
  DEVICE = torch.device("cpu")

MODEL_FILE = "models/datasetsAlpha/model_variant5_lr_0.01_n250000.pkl"

RECEIVER_TYPES = ["WR", "TE", "QB", "RB", "FB"]

def get_model_prediction_for_receiver(dataset, receiver, model_file, index):
    try:
        output_dim = 1
        x, projected_orientation, real_angle = dataset.get_orientation_based_on_receiver(receiver, index, output_dim = 1)
        x = x.float().unsqueeze(0)

        state = torch.load(model_file, map_location=DEVICE)
        model = DeepQBVariant1(input_dim=dataset.col_size - output_dim, output_dim=output_dim)
        model.load_state_dict(state)
        model.eval()

        y_hat = model(x)
        prob = torch.sigmoid(y_hat) 
        return prob.item()
    
    except RuntimeError:
        # ideal would be to change the logic to clean/prepare the data generalizing for all situations
        raise Exception("chosen instance does not have enough data (does not have data for all features used to train the model) e.g. there is no data at all for some defender or something like that to average out")

def get_frames_indexes(play_data):
    try:
        play_data.reset_index(inplace=True)
        play_data = play_data.sort_values('frameId')
        ball_snap_data = play_data[play_data["event"] == "ball_snap"]
        pass_forward_data = play_data[play_data["event"] == "pass_forward"]

        qb_frames_start = ball_snap_data.iloc[0]["frameId"] + 1
        qb_frames_end = pass_forward_data.iloc[0]["frameId"]
        
        return qb_frames_start, qb_frames_end
    except IndexError as e:
        raise Exception("Instance does not have qb pass")



def get_receiver_number(receivers, curr_receiver_id, frame_data):
    receiver_positions = frame_data[frame_data['nflId'].isin(receivers['nflId'])][['nflId', 'y']]
    receivers = receivers.merge(receiver_positions, on='nflId', how='left')
    receivers = receivers.sort_values(by='y', ascending=True).head(5)
    receivers.reset_index(inplace=True)
    receiver_row = receivers[receivers["nflId"] == curr_receiver_id]
    if receiver_row.empty:
        return -1
    return receiver_row.index[0]

def getting_angles_coordinates(projected_angles, real_angles, qb_x, qb_y, line_length):
    proj_angle_rad = np.radians(projected_angles[0])
    real_angle_rad = np.radians(real_angles[0])
    
    proj_end_x = qb_x + line_length * np.cos(proj_angle_rad)
    proj_end_y = qb_y + line_length * np.sin(proj_angle_rad)
    
    real_end_x = qb_x + line_length * np.cos(real_angle_rad)
    real_end_y = qb_y + line_length * np.sin(real_angle_rad)
    
    return proj_end_x, proj_end_y, real_end_x, real_end_y


def create_football_field() -> tuple:
    """Create a football field plot."""
    fig, ax = plt.subplots(figsize=(12, 6.33))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 53.3)
    ax.set_xlabel('Yard Line')
    ax.set_ylabel('Field Width')
    
    # Set field color
    ax.set_facecolor('forestgreen')
    
    # Add yard lines
    for x in range(10, 100, 10):
        ax.axvline(x=x, color='white', linestyle='-', linewidth=2)
        ax.text(x, 2, str(x), color='white', ha='center', fontweight='bold', fontsize=8)
    
    # Add hash marks
    for x in range(10, 100, 1):
        if x % 5 != 0:  # Skip yard lines
            ax.plot([x, x], [0, 1], 'w-', linewidth=1)  # Bottom hash
            ax.plot([x, x], [52.3, 53.3], 'w-', linewidth=1)  # Top hash
    
    # Add end zones
    ax.axvspan(0, 10, color='darkblue', alpha=0.2)
    ax.axvspan(90, 100, color='darkblue', alpha=0.2)
    
    return fig, ax

def load_data_chunked(file_path: str, chunk_size: int = 100000) -> pd.DataFrame:
    """Load data in chunks to manage memory usage."""
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # First check for NaN values in nflId
        if chunk['nflId'].isna().any():
            # Fill NaN values with a placeholder (-1)
            chunk['nflId'] = chunk['nflId'].fillna(-1)
        
        # Process each chunk with error handling
        try:
            chunk = chunk.astype({
                'playId': 'int32',
                'nflId': 'int32',
                'frameId': 'int16',
                'x': 'float32',
                'y': 'float32'
            })
        except (ValueError, TypeError) as e:
            print(f"Warning: Error during type conversion: {e}")
            print("Skipping type conversion for problematic columns")
            # Continue with original types if conversion fails
            pass
            
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

def get_play_data(game_id: int, play_id: int,
                 df_tracking: pd.DataFrame,
                 df_players: pd.DataFrame,
                 df_plays: pd.DataFrame,
                 df_games: pd.DataFrame) -> pd.DataFrame:
    """Get data for a specific play efficiently."""
    # Get tracking data for this play
    play_tracking = df_tracking[
        (df_tracking['gameId'] == game_id) & 
        (df_tracking['playId'] == play_id)
    ].copy()
    
    # Get player info
    play_players = df_players.copy()
    
    # Get play info
    play_info = df_plays[
        (df_plays['gameId'] == game_id) & 
        (df_plays['playId'] == play_id)
    ].copy()
    
    # Get game info
    game_info = df_games[df_games['gameId'] == game_id].copy()
    
    # Merge data
    play_data = pd.merge(play_tracking, play_players, on='nflId', how='left')
    play_data = pd.merge(play_data, play_info, on=['gameId', 'playId'], how='left')
    play_data = pd.merge(play_data, game_info, on='gameId', how='left')
    
    return play_data

def get_play_phases(play_data: pd.DataFrame) -> Dict[str, int]:
    """Identify key phases of the play."""
    phases = {}
    
    # Find snap frame
    snap_frames = play_data[play_data['event'] == 'ball_snap']['frameId']
    if not snap_frames.empty:
        phases['snap'] = snap_frames.iloc[0]
    
    # Find end of play frame (tackle, touchdown, etc.)
    end_events = ['tackle', 'touchdown', 'out_of_bounds', 'fumble', 'pass_outcome_incomplete']
    end_frames = play_data[play_data['event'].isin(end_events)]['frameId']
    if not end_frames.empty:
        phases['end'] = end_frames.iloc[0]
    
    return phases

def get_receiver_id(dataset, receiver_number):
    return dataset.data.iloc[0][f"nflId_{receiver_number}"]

def animate_play(game_id: int, play_id: int,
                df_tracking: pd.DataFrame,
                df_players: pd.DataFrame,
                df_plays: pd.DataFrame,
                df_games: pd.DataFrame,
                df_play_players: pd.DataFrame,
                show_labels: str = 'number',  # 'number' or 'position'
                save_path: Optional[str] = None,
                loaded=False,
                save=False,
                display_angles=False,
                line_length=15,
                receiver_to_project=0) -> None:
    """Animate player tracking data for a specific play."""

    # getting dataset for receiver passing completion analysis
    dataset = getting_frames_dataset(game_id, play_id, loaded, save, False, receiver_to_project)
    if display_angles:
        dataset_angles = getting_frames_dataset(game_id, play_id, False, False, True, receiver_to_project)

    # Get data for this play
    play_data = get_play_data(game_id, play_id, df_tracking, df_players, df_plays, df_games)

    # getting receivers data
    df_play_players = df_play_players[(df_play_players['gameId'] == game_id) & (df_play_players['playId'] == play_id)]
    receivers = df_play_players[df_play_players['routeRan'].notna()].copy()
    receivers = receivers.merge(df_players[['nflId', 'position']], on='nflId', how='left')

    # getting qb frames
    qb_frames_start, qb_frames_end = get_frames_indexes(play_data)
    range_set = set(list(range(qb_frames_start, qb_frames_end+1)))

    # Get play phases
    phases = get_play_phases(play_data)
    
    # Create football field
    fig, ax = create_football_field()
    
    # Initialize player positions and labels
    players = ax.scatter([], [], s=100)
    player_labels = []
    
    # Initialize football as a scatter point
    football = ax.scatter([], [], s=50, color='#8B4513', zorder=10)  # Saddle brown color
    
    # Initialize lines (TV-style colors)
    los = ax.axvline(x=0, color='#00A6FF', linestyle='-', alpha=0.7, linewidth=3)  # TV blue
    first_down = ax.axvline(x=0, color='#FFD700', linestyle='-', alpha=0.7, linewidth=3)  # TV yellow
    next_play = ax.axvline(x=0, color='white', linestyle='--', alpha=0.7, linewidth=2)

    angle_lines = []
    angle_texts = []
    if display_angles:
        receiver_id = get_receiver_id(dataset_angles, receiver_to_project)
        projected_line = ax.plot([], [], 'yellow', linewidth=2, alpha=0.8, label='Projected Angle')[0]
        real_line = ax.plot([], [], 'red', linewidth=2, alpha=0.8, label='Real Angle')[0]
        angle_lines = [projected_line, real_line]

    
    # Get play information
    try:
        play_info = df_plays[
            (df_plays['gameId'] == game_id) & 
            (df_plays['playId'] == play_id)
        ].iloc[0]
        
        play_description = play_info['playDescription']
        yards_gained = play_info['yardsGained']
        down = play_info['down']
        yards_to_go = play_info['yardsToGo']
        
        # Get team info
        home_team = play_info['possessionTeam']
        away_team = play_info['defensiveTeam']
        
        # Calculate line of scrimmage and first down marker
        yardline_side = play_info['yardlineSide']
        yardline_number = play_info['yardlineNumber']
        
        # Convert to absolute yard line (0-100 scale)
        # The yardline_number is how many yards from the endzone
        if yardline_side == home_team:
            los_x = yardline_number  # Already in correct format
            # First down is always in the direction of the endzone
            first_down_x = los_x + yards_to_go if yardline_number < 50 else los_x - yards_to_go
        else:
            los_x = 100 - yardline_number  # Convert from opponent's perspective
            # First down is always in the direction of the endzone
            first_down_x = los_x - yards_to_go if yardline_number < 50 else los_x + yards_to_go
            
        # Calculate next play line (if yards were gained)
        if yards_gained > 0:
            if yardline_side == home_team:
                next_play_x = los_x + yards_gained
            else:
                next_play_x = los_x - yards_gained
        else:
            next_play_x = los_x  # No gain or incomplete
            
        # Ensure lines are within field bounds
        los_x = max(0, min(100, los_x))
        first_down_x = max(0, min(100, first_down_x))
        next_play_x = max(0, min(100, next_play_x))
        
    except (KeyError, IndexError):
        los_x = 40  # Default to 40 yard line
        first_down_x = los_x + 10
        next_play_x = los_x
        play_description = "Play information not available"
        yards_gained = 0
        down = 1
        yards_to_go = 10
        home_team = 'HOME'
        away_team = 'AWAY'
    
    los.set_xdata([los_x, los_x])
    first_down.set_xdata([first_down_x, first_down_x])
    next_play.set_xdata([next_play_x, next_play_x])
    
    # Create title text
    title_text = f"Game {game_id}, Play {play_id}\n"
    title_text += f"{play_description}\n"
    title_text += f"Down: {down}, Distance: {yards_to_go} yards, Gain: {yards_gained} yards"
    print("Title: ", title_text)
    
    # Create a text object for the title
    title = ax.text(0.5, 1.05, title_text, 
                   transform=ax.transAxes,
                   ha='center', va='bottom',
                   color='white', fontweight='bold',
                   bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    
    # Create event text
    event_text = ax.text(0.5, 0.95, "", 
                        transform=ax.transAxes,
                        ha='center', va='top',
                        color='white', fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    def update(frame: int) -> List:
        """Update player positions for each frame."""
        frame_data = play_data[play_data['frameId'] == frame]
        # Update title with play phase
        phase = "Pre-snap"
        if 'snap' in phases and frame >= phases['snap']:
            phase = "Post-snap"
        if 'end' in phases and frame >= phases['end']:
            phase = "Play Complete"

        # qb data

        if display_angles:
            qb_data = frame_data[frame_data['position'] == 'QB']
            receiver_data = frame_data[frame_data['nflId'] == receiver_id].iloc[0]

            receiver_x = receiver_data['x']
            receiver_y = receiver_data['y']
            qb_x = qb_data.iloc[0]['x']
            qb_y = qb_data.iloc[0]['y']
            qb_angle = qb_data.iloc[0]['o']

            for line in angle_lines:
                line.set_data([], [])
            for text in angle_texts:
                text.remove()
            angle_texts.clear()

            # logic for more than one angle in case there is a necessity of visualizing angles of more than one receiver in the future
            projected_angles = []
            real_angles = []

            projected_orientation = dataset.get_orientation_based_on_receiver(receiver_to_project, None, 1, qb_x, qb_y, receiver_x, receiver_y)

            projected_angles.append(projected_orientation)
            real_angles.append(qb_angle)

            proj_end_x, proj_end_y, real_end_x, real_end_y =  getting_angles_coordinates(projected_angles, real_angles, qb_x, qb_y, line_length)
            angle_lines[0].set_data([qb_x, proj_end_x], [qb_y, proj_end_y])  
            angle_lines[1].set_data([qb_x, real_end_x], [qb_y, real_end_y])
            proj_text = ax.text(proj_end_x, proj_end_y, f'Proj: {projected_angles[0]:.1f}°',
                                color='yellow', fontweight='bold', fontsize=8,
                                bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
            real_text = ax.text(real_end_x, real_end_y, f'Real: {real_angles[0]:.1f}°',
                                color='red', fontweight='bold', fontsize=8,
                                bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
            
            angle_texts.extend([proj_text, real_text])



        # only getting probabilities for when the qb got the ball in hands
        probs = []

        if frame in range_set:
            for receiver in range(5):
                prob = get_model_prediction_for_receiver(dataset, receiver, MODEL_FILE, frame - qb_frames_start)
                probs.append(prob)        

        # Separate players and football
        players_data = frame_data[
            (frame_data['nflId'].notna()) & 
            (frame_data['club'].notna()) & 
            (frame_data['club'] != 'football')
        ]
        
        # Football has specific characteristics
        football_data = frame_data[
            (frame_data['nflId'].isna()) | 
            (frame_data['club'] == 'football') | 
            (frame_data['jerseyNumber'].isna())
        ]
        
        # Update scatter plot for players
        players.set_offsets(np.c_[players_data['x'] - 10, players_data['y']])
        
        # Set team colors
        colors = ['red' if team == home_team else 'blue' 
                 for team in players_data['club']]
        players.set_color(colors)
        
        # Update player labels
        for label in player_labels:
            label.remove()
        player_labels.clear()
        
        title.set_text(f"{title_text}\nPhase: {phase}")

        for idx, row in players_data.iterrows():
            if show_labels == 'number':
                try:
                    jersey_num = int(row['jerseyNumber'])
                    label_text = str(jersey_num)
                except (ValueError, TypeError):
                    label_text = '?'
            else: 
                label_text = row['position']
                if label_text in RECEIVER_TYPES and frame in range_set:
                    receiver_number = get_receiver_number(receivers, row["nflId"], frame_data)
                    if receiver_number != -1:
                        label_text = round(probs[receiver_number] * 100, 2) 
            
            label = ax.text(row['x'] - 10, row['y'], label_text,
                          color='white', ha='center', va='center',
                          fontsize=6, fontweight='bold',
                          path_effects=[withStroke(linewidth=2, foreground='black')])
            player_labels.append(label)
        
        # Update football position
        if not football_data.empty:
            football.set_offsets(np.c_[football_data['x'].iloc[0] - 10, football_data['y'].iloc[0]])
            football.set_visible(True)
        else:
            football.set_visible(False)
                
        # Update event text
        current_event = frame_data['event'].iloc[0] if not frame_data['event'].isna().all() else ""
        if current_event:
            event_text.set_text(f"Event: {current_event}")
        else:
            event_text.set_text("")
        
        return_list = [players, football, los, first_down, next_play, title, event_text] + player_labels
        if display_angles:
            return_list.extend(angle_lines)
            return_list.extend(angle_texts)

        return return_list
    
    # Create animation
    ani = animation.FuncAnimation(fig, 
                                update,
                                frames=play_data['frameId'].unique(),
                                interval=100,
                                blit=True)
    
    if save_path:
        ani.save(save_path, writer='pillow', fps=10)
    else:
        plt.show()

def main():
    # Load data in chunks
    print("Loading tracking data...")
    df_tracking = load_data_chunked('data/tracking_week_1.csv')
    
    print("Loading player data...")
    df_players = pd.read_csv('data/players.csv')
    
    print("Loading play data...")
    df_plays = pd.read_csv('data/plays.csv')
    
    print("Loading game data...")
    df_games = pd.read_csv('data/games.csv')

    print("Loading game data...")
    df_play_players = pd.read_csv('data/player_play.csv')
    
    # Example game and play IDs
    # Have data complete for sure game_id: 2022091200, play_id: 2688:
    game_id = 2022091200
    play_id = 1487
    #  64,   85,  109,  156,  180,  201,  264,  286,  315,  346,  375,
    #     401,  446,  467,  565,  601,  622,  643,  664,  688,  716,  741,
    #     762,  786,  810,  882,  910,  931,  983, 1004, 1028, 1057, 1092,
    #    1164, 1217, 1241, 1299, 1320, 1344, 1409, 1433, 1465, 1487, 1521,
    #    1550, 1579, 1642, 1680, 1704, 1725, 1764, 1793, 1815, 1851, 1967,
    #    1988, 2009, 2038, 2067, 2093, 2188, 2244, 2268, 2292, 2370, 2391,
    #    2479, 2500, 2522, 2546, 2591, 2613, 2667, 2688, 2712, 2750, 2779,
    #    2801, 2830, 2883, 2923, 2944, 2965, 3001, 3048, 3077, 3101, 3125,
    #    3149, 3173, 3194, 3216, 3245, 3267, 3296, 3325, 3382, 3404, 3467,
    #    3491, 3515, 3553, 3574, 3596, 3628, 3685, 3723, 3747, 3795, 3826,
    #    3980, 4012
    
    print(f"Animating game {game_id}, play {play_id}...")
    animate_play(game_id, play_id, df_tracking, df_players, df_plays, df_games, df_play_players, show_labels='position', loaded=False, save=True, display_angles=False)

if __name__ == "__main__":
    main() 