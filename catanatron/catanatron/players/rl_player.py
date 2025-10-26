import torch
import numpy as np
from typing import Tuple

from catanatron.game import Game
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.features import create_sample_vector
from catanatron.gym.board_tensor_features import create_board_tensor
from rl.models import ValueNetwork

def game_to_features(game: Game, color: int) -> np.ndarray:
    """
    Extracts features from the game state for the value network.
    """
    # This function creates a fixed-order vector of features from the game state.
    feature_vector = create_sample_vector(game, color)
    board_tensor = create_board_tensor(game, color)
    return np.concatenate([feature_vector, board_tensor.flatten()], axis=0)


class RLValuePlayer(AlphaBetaPlayer):
    """
    An AlphaBetaPlayer that uses a trained ValueNetwork for leaf evaluation.
    """

    def __init__(
        self,
        color,
        model_path: str,
        input_dim: int,
        output_range: Tuple[float, float] = (-1, 1),
        **kwargs
    ):
        super().__init__(color, **kwargs)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load the trained value network
        self.value_net = ValueNetwork(input_dim).to(self.device)
        self.value_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.value_net.eval()

        self.use_value_function = True
        self.value_fn_builder_name = "rl_value_network"

    def value_function(self, game: Game, p0_color: int) -> float:
        """
        Overrides the heuristic value function to use the neural network.
        """
        game_tensor = torch.from_numpy(game_to_features(game, p0_color).reshape(1, -1).astype(np.float32)).to(self.device)
        with torch.no_grad():
            value = self.value_net(game_tensor).item()

        # Clip the value to the output range
        if value < self.output_range[0]:
            value = self.output_range[0]
        elif value > self.output_range[1]:
            value = self.output_range[1]
        return value

    def __repr__(self) -> str:
        return super().__repr__()
