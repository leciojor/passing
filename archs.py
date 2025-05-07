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

        #First layer 230 -> 256
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

        # softmax
        logits = self.output_layer(x)
        probs = F.softmax(logits, dim=1)
        return probs
    

# ---------------------------------------------------
# DeepQB Variant 2: Predict expected yards gained
# INPUT
#   containing play context, QB state, and each WR/defender state
# OUTPUT
#   5 continuous values, one per eligible receiver (expected yardage)
# ---------------------------------------------------

class DeepQBVariant2(nn.Module):
    def __init__(self, input_dim=230, hidden_dim1=256, hidden_dim2=128, output_dim=5, dropout_rate=0.3):
        super(DeepQBVariant2, self).__init__()

        # First layer 230 -> 256
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second layer 256 -> 128
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Output layer -> 5
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

        # regression
        expected_yards = self.output_layer(x)
        return expected_yards



# ---------------------------------------------------
# DeepQB Variant 3: Predict pass outcome probabilities
# INPUT
#   containing play context, QB state, and each WR/defender state
# OUTPUT
#   probability distribution over 3 pass outcomes:
#     0 - Completion, 1 - Incompletion, 2 - Interception
# ---------------------------------------------------

class DeepQBVariant3(nn.Module):
    def __init__(self, input_dim=230, hidden_dim1=256, hidden_dim2=128, output_dim=3, dropout_rate=0.3):
        super(DeepQBVariant3, self).__init__()

        # First layer 230 -> 256
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second layer 256 -> 128
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Output layer ->  0 - Completion, 1 - Incompletion, 2 - Interception
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

        # Softmax  0 - Completion, 1 - Incompletion, 2 - Interception
        logits = self.output_layer(x)
        probs = F.softmax(logits, dim=1)
        return probs