#include "cpptanatron/core.hpp"

#include <algorithm>
#include <queue>

namespace cpptanatron {
namespace {

constexpr std::array<std::array<int, 3>, kNumNodes> kAdjacency = {{
    {{1, 5, 20}},    {{0, 2, 6}},     {{1, 3, 9}},
    {{2, 4, 12}},    {{3, 5, 15}},    {{0, 4, 16}},
    {{1, 7, 23}},    {{6, 8, 24}},    {{7, 9, 27}},
    {{2, 8, 10}},    {{9, 11, 29}},   {{10, 12, 32}},
    {{3, 11, 13}},   {{12, 14, 34}},  {{13, 15, 37}},
    {{4, 14, 17}},   {{5, 18, 21}},   {{15, 18, 39}},
    {{16, 17, 40}},  {{20, 21, 46}},  {{0, 19, 22}},
    {{16, 19, 43}},  {{20, 23, 49}},  {{6, 22, 52}},
    {{7, 25, 53}},   {{24, 26, -1}},  {{25, 27, -1}},
    {{8, 26, 28}},   {{27, 29, -1}},  {{10, 28, 30}},
    {{29, 31, -1}},  {{30, 32, -1}},  {{11, 31, 33}},
    {{32, 34, -1}},  {{13, 33, 35}},  {{34, 36, -1}},
    {{35, 37, -1}},  {{14, 36, 38}},  {{37, 39, -1}},
    {{17, 38, 41}},  {{18, 42, 44}},  {{39, 42, -1}},
    {{40, 41, -1}},  {{21, 44, 47}},  {{40, 43, -1}},
    {{46, 47, -1}},  {{19, 45, 48}},  {{43, 45, -1}},
    {{46, 49, -1}},  {{22, 48, 50}},  {{49, 51, -1}},
    {{50, 52, -1}},  {{23, 51, 53}},  {{24, 52, -1}},
}};

constexpr std::array<Edge, kNumEdges> kEdges = {{
    {0, 1},   {0, 5},   {0, 20},  {1, 2},   {1, 6},   {2, 3},
    {2, 9},   {3, 4},   {3, 12},  {4, 5},   {4, 15},  {5, 16},
    {6, 7},   {6, 23},  {7, 8},   {7, 24},  {8, 9},   {8, 27},
    {9, 10},  {10, 11}, {10, 29}, {11, 12}, {11, 32}, {12, 13},
    {13, 14}, {13, 34}, {14, 15}, {14, 37}, {15, 17}, {16, 18},
    {16, 21}, {17, 18}, {17, 39}, {18, 40}, {19, 20}, {19, 21},
    {19, 46}, {20, 22}, {21, 43}, {22, 23}, {22, 49}, {23, 52},
    {24, 25}, {24, 53}, {25, 26}, {26, 27}, {27, 28}, {28, 29},
    {29, 30}, {30, 31}, {31, 32}, {32, 33}, {33, 34}, {34, 35},
    {35, 36}, {36, 37}, {37, 38}, {38, 39}, {39, 41}, {40, 42},
    {40, 44}, {41, 42}, {43, 44}, {43, 47}, {45, 46}, {45, 47},
    {46, 48}, {48, 49}, {49, 50}, {50, 51}, {51, 52}, {52, 53},
}};

int edge_index(Edge edge) {
  edge = edge.normalized();
  for (int i = 0; i < kNumEdges; ++i) {
    if (kEdges[i] == edge) {
      return i;
    }
  }
  return -1;
}

bool adjacent(int a, int b) {
  if (a < 0 || a >= kNumNodes || b < 0 || b >= kNumNodes) {
    return false;
  }
  for (int n : kAdjacency[a]) {
    if (n == b) {
      return true;
    }
  }
  return false;
}

}  // namespace

Edge Edge::normalized() const {
  return a <= b ? *this : Edge{b, a};
}

bool Edge::operator==(const Edge& other) const {
  return a == other.a && b == other.b;
}

Action Action::build_settlement(Color color, int node_id) {
  return Action{color, ActionType::BuildSettlement, node_id, -1};
}

Action Action::build_road(Color color, int node_a, int node_b) {
  Edge edge{node_a, node_b};
  edge = edge.normalized();
  return Action{color, ActionType::BuildRoad, edge.a, edge.b};
}

Action Action::end_turn(Color color) {
  return Action{color, ActionType::EndTurn, -1, -1};
}

Board::Board() {
  node_owner_.fill(kNoOwner);
  node_building_.fill(static_cast<int>(BuildingType::Empty));
  board_buildable_.fill(1);
  edge_owner_.fill(kNoOwner);
}

int Board::color_index(Color color) {
  return static_cast<int>(color);
}

bool Board::is_empty_node(int node_id) const {
  return node_id >= 0 && node_id < kNumNodes && node_owner_[node_id] == kNoOwner;
}

bool Board::has_road(Color color, Edge edge) const {
  int idx = edge_index(edge);
  return idx >= 0 && edge_owner_[idx] == color_index(color);
}

bool Board::is_edge_occupied(Edge edge) const {
  int idx = edge_index(edge);
  return idx >= 0 && edge_owner_[idx] != kNoOwner;
}

void Board::build_settlement(Color color, int node_id, bool initial_build_phase) {
  auto buildable = buildable_node_ids(color, initial_build_phase);
  if (std::find(buildable.begin(), buildable.end(), node_id) == buildable.end()) {
    throw std::invalid_argument("Invalid settlement placement");
  }
  if (!is_empty_node(node_id)) {
    throw std::invalid_argument("A building already exists at this node");
  }

  node_owner_[node_id] = color_index(color);
  node_building_[node_id] = static_cast<int>(BuildingType::Settlement);
  board_buildable_[node_id] = 0;
  for (int neighbor : kAdjacency[node_id]) {
    if (neighbor >= 0) {
      board_buildable_[neighbor] = 0;
    }
  }
}

void Board::build_road(Color color, Edge edge) {
  edge = edge.normalized();
  auto buildable = buildable_edges(color);
  if (std::find(buildable.begin(), buildable.end(), edge) == buildable.end()) {
    throw std::invalid_argument("Invalid road placement");
  }

  int idx = edge_index(edge);
  edge_owner_[idx] = color_index(color);
}

std::vector<int> Board::buildable_node_ids(Color color, bool initial_build_phase) const {
  std::vector<int> result;
  if (initial_build_phase) {
    for (int node = 0; node < kNumNodes; ++node) {
      if (board_buildable_[node]) {
        result.push_back(node);
      }
    }
    return result;
  }

  for (int node = 0; node < kNumNodes; ++node) {
    if (!board_buildable_[node]) {
      continue;
    }
    for (int neighbor : kAdjacency[node]) {
      if (neighbor >= 0 && has_road(color, Edge{node, neighbor})) {
        result.push_back(node);
        break;
      }
    }
  }
  return result;
}

std::vector<Edge> Board::buildable_edges(Color color) const {
  std::vector<Edge> result;
  for (const Edge& edge : kEdges) {
    if (is_edge_occupied(edge)) {
      continue;
    }
    bool touches_owned_node =
        node_owner_[edge.a] == color_index(color) || node_owner_[edge.b] == color_index(color);
    bool touches_owned_road = false;
    for (int neighbor : kAdjacency[edge.a]) {
      touches_owned_road = touches_owned_road || (neighbor >= 0 && has_road(color, Edge{edge.a, neighbor}));
    }
    for (int neighbor : kAdjacency[edge.b]) {
      touches_owned_road = touches_owned_road || (neighbor >= 0 && has_road(color, Edge{edge.b, neighbor}));
    }
    if (touches_owned_node || touches_owned_road) {
      result.push_back(edge);
    }
  }
  return result;
}

std::array<int, kNumEdges> Board::edge_owner_by_index() const {
  return edge_owner_;
}

State::State(std::vector<Player> players_in) : players(std::move(players_in)) {
  if (players.empty() || players.size() > kMaxPlayers) {
    throw std::invalid_argument("Game requires 1 to 4 players");
  }
  for (const auto& player : players) {
    colors.push_back(player.color);
  }
  victory_points.fill(0);
  roads_available.fill(15);
  settlements_available.fill(5);
  cities_available.fill(4);
  last_settlement_node.fill(-1);
}

Color State::current_color() const {
  return colors.at(current_player_index);
}

const Player& State::current_player() const {
  return players.at(current_player_index);
}

void State::advance_turn(int delta) {
  const int n = static_cast<int>(colors.size());
  current_player_index = (current_player_index + delta + n) % n;
  current_turn_index = current_player_index;
  if (delta > 0) {
    num_turns += 1;
  }
}

Game::Game(std::vector<Player> players, int vps) : state(std::move(players)), vps_to_win(vps) {}

std::vector<Action> Game::generate_playable_actions() const {
  const Color color = state.current_color();
  std::vector<Action> actions;
  if (state.current_prompt == ActionPrompt::BuildInitialSettlement) {
    for (int node : state.board.buildable_node_ids(color, true)) {
      actions.push_back(Action::build_settlement(color, node));
    }
  } else if (state.current_prompt == ActionPrompt::BuildInitialRoad) {
    const int last_settlement = state.last_settlement_node[state.current_player_index];
    for (const Edge& edge : state.board.buildable_edges(color)) {
      if (edge.a == last_settlement || edge.b == last_settlement) {
        actions.push_back(Action::build_road(color, edge.a, edge.b));
      }
    }
  } else if (state.current_prompt == ActionPrompt::PlayTurn) {
    actions.push_back(Action{color, ActionType::Roll, -1, -1});
  }
  return actions;
}

ActionRecord Game::execute(const Action& action, bool validate_action) {
  if (validate_action) {
    const auto playable = generate_playable_actions();
    const bool valid = std::any_of(playable.begin(), playable.end(), [&](const Action& option) {
      return option.color == action.color && option.type == action.type &&
             option.value0 == action.value0 && option.value1 == action.value1;
    });
    if (!valid) {
      throw std::invalid_argument("Action is not playable right now");
    }
  }

  if (action.type == ActionType::BuildSettlement) {
    state.board.build_settlement(action.color, action.value0, state.is_initial_build_phase);
    const int player = state.current_player_index;
    state.settlements_available[player] -= 1;
    state.victory_points[player] += 1;
    state.last_settlement_node[player] = action.value0;
    state.current_prompt = ActionPrompt::BuildInitialRoad;
  } else if (action.type == ActionType::BuildRoad) {
    state.board.build_road(action.color, Edge{action.value0, action.value1});
    const int player = state.current_player_index;
    state.roads_available[player] -= 1;

    if (state.is_initial_build_phase) {
      const int num_players = static_cast<int>(state.colors.size());
      int total_settlements = 0;
      for (int i = 0; i < num_players; ++i) {
        total_settlements += 5 - state.settlements_available[i];
      }
      if (total_settlements < num_players) {
        state.advance_turn(1);
        state.current_prompt = ActionPrompt::BuildInitialSettlement;
      } else if (total_settlements == num_players) {
        state.current_prompt = ActionPrompt::BuildInitialSettlement;
      } else if (total_settlements == 2 * num_players) {
        state.is_initial_build_phase = false;
        state.current_prompt = ActionPrompt::PlayTurn;
      } else {
        state.advance_turn(-1);
        state.current_prompt = ActionPrompt::BuildInitialSettlement;
      }
    }
  } else if (action.type == ActionType::EndTurn) {
    state.advance_turn(1);
    state.current_prompt = ActionPrompt::PlayTurn;
  } else {
    throw std::invalid_argument("Action type not implemented in the C++ parity core yet");
  }

  ActionRecord record{action, -1, -1};
  state.action_records.push_back(record);
  return record;
}

Snapshot Game::snapshot() const {
  Snapshot snapshot;
  for (Color color : state.colors) {
    snapshot.colors.push_back(static_cast<int>(color));
  }
  snapshot.current_player_index = state.current_player_index;
  snapshot.current_turn_index = state.current_turn_index;
  snapshot.current_prompt = state.current_prompt;
  snapshot.is_initial_build_phase = state.is_initial_build_phase;
  snapshot.num_turns = state.num_turns;
  snapshot.victory_points = state.victory_points;
  snapshot.roads_available = state.roads_available;
  snapshot.settlements_available = state.settlements_available;
  snapshot.cities_available = state.cities_available;
  snapshot.node_owner = state.board.node_owner();
  snapshot.node_building = state.board.node_building();
  snapshot.edge_owner = state.board.edge_owner_by_index();
  return snapshot;
}

std::vector<Edge> static_edges() {
  return std::vector<Edge>(kEdges.begin(), kEdges.end());
}

std::string to_string(Color color) {
  switch (color) {
    case Color::Red:
      return "RED";
    case Color::Blue:
      return "BLUE";
    case Color::Orange:
      return "ORANGE";
    case Color::White:
      return "WHITE";
  }
  return "UNKNOWN";
}

std::string to_string(ActionPrompt prompt) {
  switch (prompt) {
    case ActionPrompt::BuildInitialSettlement:
      return "BUILD_INITIAL_SETTLEMENT";
    case ActionPrompt::BuildInitialRoad:
      return "BUILD_INITIAL_ROAD";
    case ActionPrompt::PlayTurn:
      return "PLAY_TURN";
    case ActionPrompt::Discard:
      return "DISCARD";
    case ActionPrompt::MoveRobber:
      return "MOVE_ROBBER";
    case ActionPrompt::DecideTrade:
      return "DECIDE_TRADE";
    case ActionPrompt::DecideAcceptees:
      return "DECIDE_ACCEPTEES";
  }
  return "UNKNOWN";
}

std::string to_string(ActionType action_type) {
  switch (action_type) {
    case ActionType::Roll:
      return "ROLL";
    case ActionType::MoveRobber:
      return "MOVE_ROBBER";
    case ActionType::DiscardResource:
      return "DISCARD_RESOURCE";
    case ActionType::BuildRoad:
      return "BUILD_ROAD";
    case ActionType::BuildSettlement:
      return "BUILD_SETTLEMENT";
    case ActionType::BuildCity:
      return "BUILD_CITY";
    case ActionType::BuyDevelopmentCard:
      return "BUY_DEVELOPMENT_CARD";
    case ActionType::PlayKnightCard:
      return "PLAY_KNIGHT_CARD";
    case ActionType::PlayYearOfPlenty:
      return "PLAY_YEAR_OF_PLENTY";
    case ActionType::PlayMonopoly:
      return "PLAY_MONOPOLY";
    case ActionType::PlayRoadBuilding:
      return "PLAY_ROAD_BUILDING";
    case ActionType::MaritimeTrade:
      return "MARITIME_TRADE";
    case ActionType::OfferTrade:
      return "OFFER_TRADE";
    case ActionType::AcceptTrade:
      return "ACCEPT_TRADE";
    case ActionType::RejectTrade:
      return "REJECT_TRADE";
    case ActionType::ConfirmTrade:
      return "CONFIRM_TRADE";
    case ActionType::CancelTrade:
      return "CANCEL_TRADE";
    case ActionType::EndTurn:
      return "END_TURN";
  }
  return "UNKNOWN";
}

}  // namespace cpptanatron
