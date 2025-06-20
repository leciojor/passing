import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------
# DeepQB Variant 1: Predict the target receiver
# INPUT
#   containing play context, QB state, and each WR/defender state
# OUTPUT
#   probability distribution over 5 receivers
# ---------------------------------------------------

class DeepQBVariant1(nn.Module):
    def __init__(self, input_dim=230, hidden_dim1=256, hidden_dim2=128, output_dim=5, dropout_rate=0.3):
        super(DeepQBVariant1, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second layer 256 -> 128
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Output layer -> 5 WR
        self.output_layer = nn.Linear(hidden_dim2, output_dim)


    def forward(self, x):
        # Hidden Layer 1
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.dropout1(x)

        # Hidden Layer 2
        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.dropout2(x)

        logits = self.output_layer(x)
        # probs = F.softmax(logits, dim=1)
        return logits

#logic to get the intended receiver based exclusively on the projected oriented that is the closest to the real orientation
def intended_receiver_simple_algo(dataset, index):
        diff = float("inf")
        prediction = None
        for _ in range(5):
            projected_orientation, real_angle, intended = dataset.get_orientation_based_on_receiver(_, index, intended=True)
            curr_diff = abs((projected_orientation - real_angle + 180) % 360 - 180)
            if curr_diff < diff:
                diff = curr_diff
                prediction = _

        return prediction, intended

def getting_all_intended_receivers_simple_algo(dataset, index):
    predictions, intended_receivers  = [], []

    for i in range(len(dataset)):
        if not index is None:
            prediction, intended = intended_receiver_simple_algo(dataset, index)
        else:
            prediction, intended = intended_receiver_simple_algo(dataset, i)

        if prediction and intended:
            predictions.append(prediction)
            intended_receivers.append(intended)
        
        if not index is None:
            break
    return predictions, intended_receivers


# ---------------------------------------------------
# DeepQB Variant 2: Predict expected yards gained
# INPUT
#   containing play context, QB state, and each WR/defender state
# OUTPUT
#   5 continuous values, one per eligible receiver (expected yardage)
# ---------------------------------------------------


# ---------------------------------------------------
# DeepQB Variant 3: Predict pass outcome probabilities
# INPUT
#   containing play context, QB state, and each WR/defender state
# OUTPUT
#   probability distribution over 3 pass outcomes:
#     0 - Completion, 1 - Incompletion, 2 - Interception
# ---------------------------------------------------

