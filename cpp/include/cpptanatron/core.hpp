#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cpptanatron {

constexpr int kMaxPlayers = 4;
constexpr int kNumNodes = 54;
constexpr int kNumEdges = 72;
constexpr int kNoOwner = -1;

enum class Color : std::int8_t {
  Red = 0,
  Blue = 1,
  Orange = 2,
  White = 3,
};

enum class ActionPrompt : std::int8_t {
  BuildInitialSettlement = 0,
  BuildInitialRoad = 1,
  PlayTurn = 2,
  Discard = 3,
  MoveRobber = 4,
  DecideTrade = 5,
  DecideAcceptees = 6,
};

enum class ActionType : std::int8_t {
  Roll = 0,
  MoveRobber = 1,
  DiscardResource = 2,
  BuildRoad = 3,
  BuildSettlement = 4,
  BuildCity = 5,
  BuyDevelopmentCard = 6,
  PlayKnightCard = 7,
  PlayYearOfPlenty = 8,
  PlayMonopoly = 9,
  PlayRoadBuilding = 10,
  MaritimeTrade = 11,
  OfferTrade = 12,
  AcceptTrade = 13,
  RejectTrade = 14,
  ConfirmTrade = 15,
  CancelTrade = 16,
  EndTurn = 17,
};

enum class BuildingType : std::int8_t {
  Empty = 0,
  Settlement = 1,
  City = 2,
};

struct Edge {
  int a;
  int b;

  Edge normalized() const;
  bool operator==(const Edge& other) const;
};

struct Action {
  Color color;
  ActionType type;
  int value0 = -1;
  int value1 = -1;

  static Action build_settlement(Color color, int node_id);
  static Action build_road(Color color, int node_a, int node_b);
  static Action end_turn(Color color);
};

struct ActionRecord {
  Action action;
  int result0 = -1;
  int result1 = -1;
};

struct Player {
  Color color;
  bool is_bot = true;
};

struct Snapshot {
  std::vector<int> colors;
  int current_player_index = 0;
  int current_turn_index = 0;
  ActionPrompt current_prompt = ActionPrompt::BuildInitialSettlement;
  bool is_initial_build_phase = true;
  int num_turns = 0;
  std::array<int, kMaxPlayers> victory_points{};
  std::array<int, kMaxPlayers> roads_available{};
  std::array<int, kMaxPlayers> settlements_available{};
  std::array<int, kMaxPlayers> cities_available{};
  std::array<int, kNumNodes> node_owner{};
  std::array<int, kNumNodes> node_building{};
  std::array<int, kNumEdges> edge_owner{};
};

class Board {
 public:
  Board();

  void build_settlement(Color color, int node_id, bool initial_build_phase);
  void build_road(Color color, Edge edge);

  std::vector<int> buildable_node_ids(Color color, bool initial_build_phase) const;
  std::vector<Edge> buildable_edges(Color color) const;

  const std::array<int, kNumNodes>& node_owner() const { return node_owner_; }
  const std::array<int, kNumNodes>& node_building() const { return node_building_; }
  std::array<int, kNumEdges> edge_owner_by_index() const;

 private:
  std::array<int, kNumNodes> node_owner_{};
  std::array<int, kNumNodes> node_building_{};
  std::array<int, kNumNodes> board_buildable_{};
  std::array<int, kNumEdges> edge_owner_{};

  static int color_index(Color color);
  bool is_empty_node(int node_id) const;
  bool has_road(Color color, Edge edge) const;
  bool is_edge_occupied(Edge edge) const;
};

class State {
 public:
  explicit State(std::vector<Player> players);

  Color current_color() const;
  const Player& current_player() const;
  void advance_turn(int delta = 1);

  Board board;
  std::vector<Player> players;
  std::vector<Color> colors;
  int current_player_index = 0;
  int current_turn_index = 0;
  ActionPrompt current_prompt = ActionPrompt::BuildInitialSettlement;
  bool is_initial_build_phase = true;
  int num_turns = 0;
  std::array<int, kMaxPlayers> victory_points{};
  std::array<int, kMaxPlayers> roads_available{};
  std::array<int, kMaxPlayers> settlements_available{};
  std::array<int, kMaxPlayers> cities_available{};
  std::array<int, kMaxPlayers> last_settlement_node{};
  std::vector<ActionRecord> action_records;
};

class Game {
 public:
  explicit Game(std::vector<Player> players, int vps_to_win = 10);

  ActionRecord execute(const Action& action, bool validate_action = true);
  std::vector<Action> generate_playable_actions() const;
  Snapshot snapshot() const;

  State state;
  int vps_to_win = 10;
};

std::vector<Edge> static_edges();
std::string to_string(Color color);
std::string to_string(ActionPrompt prompt);
std::string to_string(ActionType action_type);

}  // namespace cpptanatron
