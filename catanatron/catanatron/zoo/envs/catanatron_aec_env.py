from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import AECEnv

from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.player import Color, Player
from catanatron.models.map import build_map
from catanatron.features import create_sample, get_feature_ordering
from catanatron.gym.board_tensor_features import (
    create_board_tensor,
    get_channels,
    is_graph_feature,
)
from catanatron.gym.envs.catanatron_env import (
    ACTION_SPACE_SIZE,
    to_action_space,
    from_action_space,
)


HIGH = 19 * 5


def _color_order_for_num_players(num_players: int) -> List[Color]:
    all_colors = [Color.BLUE, Color.RED, Color.WHITE, Color.ORANGE]
    assert 2 <= num_players <= 4, "num_players must be in [2, 4]"
    return all_colors[:num_players]


class CatanatronAECEnv(AECEnv):  # type: ignore[misc]
    """
    PettingZoo AECEnv for multi-agent Catanatron (turn-based).

    - Only the current agent (according to game state) acts per step.
    - Observations can be 'vector' or 'mixed' (board tensor + numeric).
    - Rewards: winner 1, others -1 on termination; 0 otherwise.
      Invalid actions yield `invalid_action_reward`; exceeding `max_invalid_actions`
      truncates the episode.
    """

    metadata = {"name": "catanatron_aec_v0", "render_modes": []}

    def __init__(self, config: Optional[Dict] = None):
        self.config = (config or {}).copy()

        self.num_players: int = int(self.config.get("num_players", 2))
        assert 2 <= self.num_players <= 4, "num_players must be in [2, 4]"
        self.map_type: str = self.config.get("map_type", "BASE")
        self.vps_to_win: int = int(self.config.get("vps_to_win", 10))
        self.representation: str = self.config.get("representation", "vector")
        assert self.representation in ["vector", "mixed"]
        self.invalid_action_reward: float = float(
            self.config.get("invalid_action_reward", -1.0)
        )
        self.max_invalid_actions: int = int(self.config.get("max_invalid_actions", 10))

        # Agents/color mapping
        self.possible_agents: List[str] = [f"player_{i}" for i in range(self.num_players)]
        self.agents: List[str] = []
        self.colors_order: List[Color] = _color_order_for_num_players(self.num_players)
        self.agent_name_to_color: Dict[str, Color] = {
            f"player_{i}": color for i, color in enumerate(self.colors_order)
        }
        self.color_to_agent_name: Dict[Color, str] = {
            color: f"player_{i}" for i, color in enumerate(self.colors_order)
        }

        # Features and observation/action spaces
        self.features: List[str] = get_feature_ordering(self.num_players, self.map_type)
        if self.representation == "mixed":
            channels = get_channels(self.num_players)
            self._board_tensor_shape: Tuple[int, int, int] = (channels, 21, 11)
            self.numeric_features: List[str] = [
                f for f in self.features if not is_graph_feature(f)
            ]
            obs_space = spaces.Dict(
                {
                    "board": spaces.Box(
                        low=0, high=1, shape=self._board_tensor_shape, dtype=np.float64
                    ),
                    "numeric": spaces.Box(
                        low=0,
                        high=HIGH,
                        shape=(len(self.numeric_features),),
                        dtype=np.float64,
                    ),
                }
            )
        else:
            obs_space = spaces.Box(
                low=0, high=HIGH, shape=(len(self.features),), dtype=np.float64
            )

        self.observation_spaces: Dict[str, spaces.Space] = {
            agent: obs_space for agent in self.possible_agents
        }
        self.action_spaces: Dict[str, spaces.Space] = {
            agent: spaces.Discrete(ACTION_SPACE_SIZE) for agent in self.possible_agents
        }

        # Runtime
        self.game: Optional[Game] = None
        self.agent_selection: Optional[str] = None
        self.rewards: Dict[str, float] = {}
        self._cumulative_rewards: Dict[str, float] = {}
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.infos: Dict[str, Dict] = {}
        self._invalid_actions_count: Dict[str, int] = {}

    # AEC API
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        catan_map = build_map(self.map_type)
        players = [Player(color) for color in self.colors_order]
        for p in players:
            p.reset_state()
        self.game = Game(
            players=players, seed=seed, catan_map=catan_map, vps_to_win=self.vps_to_win
        )

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._invalid_actions_count = {agent: 0 for agent in self.agents}

        self.agent_selection = self._current_agent()
        self._update_infos_for_current()

    def observe(self, agent: str) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        assert self.game is not None
        color = self.agent_name_to_color[agent]
        sample = create_sample(self.game, color)
        if self.representation == "mixed":
            board_tensor = create_board_tensor(
                self.game, color, channels_first=True
            )
            numeric = np.array([float(sample[i]) for i in self.numeric_features])
            return {"board": board_tensor, "numeric": numeric}
        return np.array([float(sample[i]) for i in self.features])

    def step(self, action: int):
        assert self.game is not None, "reset() must be called before step()"
        if self.agent_selection is None or len(self.agents) == 0:
            return

        agent = self.agent_selection

        if self.terminations[agent] or self.truncations[agent]:
            # Required no-op step for done agents
            self._clear_rewards()
            self._accumulate_rewards()
            self.agent_selection = self._next_agent_if_any()
            return

        # Clear step rewards
        self._clear_rewards()

        # Execute action for the current agent
        current_color = self.game.state.current_color()
        current_agent = self.color_to_agent_name[current_color]
        if agent != current_agent:
            # If external agent tries to act out of turn, penalize and ignore
            self._invalid_actions_count[agent] += 1
            self.rewards[agent] = self.invalid_action_reward
        else:
            try:
                catan_action = from_action_space(action, self.game.state.playable_actions)
                self.game.execute(catan_action)
            except Exception:
                self._invalid_actions_count[agent] += 1
                self.rewards[agent] = self.invalid_action_reward

        # Check termination/truncation
        winning_color = self.game.winning_color()
        if winning_color is not None:
            for color in self.colors_order:
                ag = self.color_to_agent_name[color]
                self.terminations[ag] = True
                self.rewards[ag] = 1.0 if color == winning_color else -1.0
            self.agents = []
        elif self.game.state.num_turns >= TURNS_LIMIT:
            for ag in list(self.truncations.keys()):
                self.truncations[ag] = True
            self.agents = []
        elif any(
            count > self.max_invalid_actions for count in self._invalid_actions_count.values()
        ):
            for ag in list(self.truncations.keys()):
                self.truncations[ag] = True
            self.agents = []
        else:
            # Continue: select next agent based on game state
            self.agent_selection = self._current_agent()
            self._update_infos_for_current()

        # Update cumulative rewards for wrappers
        self._accumulate_rewards()

    # Minimal rendering/closing (headless environment)
    def render(self):
        return None

    def close(self):
        return None

    # Helpers
    def _current_agent(self) -> str:
        assert self.game is not None
        current_color = self.game.state.current_color()
        return self.color_to_agent_name[current_color]

    def _next_agent_if_any(self) -> Optional[str]:
        if len(self.agents) == 0:
            return None
        return self._current_agent()

    def _clear_rewards(self):
        for ag in self.rewards:
            self.rewards[ag] = 0.0

    def _update_infos_for_current(self):
        assert self.game is not None
        for ag in self.infos:
            self.infos[ag] = {}
        valid_actions = list(map(to_action_space, self.game.state.playable_actions))
        current_agent = self._current_agent()
        self.infos[current_agent] = {"valid_actions": valid_actions}


def aec_env(config: Optional[Dict] = None) -> CatanatronAECEnv:
    return CatanatronAECEnv(config=config)