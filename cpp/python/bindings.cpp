#include "cpptanatron/core.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace cpptanatron;

namespace {

template <typename T, std::size_t N>
std::vector<T> to_vector(const std::array<T, N>& values) {
  return std::vector<T>(values.begin(), values.end());
}

py::dict snapshot_to_dict(const Snapshot& snapshot) {
  py::dict result;
  result["colors"] = snapshot.colors;
  result["current_player_index"] = snapshot.current_player_index;
  result["current_turn_index"] = snapshot.current_turn_index;
  result["current_prompt"] = to_string(snapshot.current_prompt);
  result["current_prompt_id"] = static_cast<int>(snapshot.current_prompt);
  result["is_initial_build_phase"] = snapshot.is_initial_build_phase;
  result["num_turns"] = snapshot.num_turns;
  result["victory_points"] = to_vector(snapshot.victory_points);
  result["roads_available"] = to_vector(snapshot.roads_available);
  result["settlements_available"] = to_vector(snapshot.settlements_available);
  result["cities_available"] = to_vector(snapshot.cities_available);
  result["node_owner"] = to_vector(snapshot.node_owner);
  result["node_building"] = to_vector(snapshot.node_building);
  result["edge_owner"] = to_vector(snapshot.edge_owner);
  return result;
}

}  // namespace

PYBIND11_MODULE(_cpp_engine, m) {
  m.doc() = "C++ Catanatron parity core";

  py::enum_<Color>(m, "Color")
      .value("RED", Color::Red)
      .value("BLUE", Color::Blue)
      .value("ORANGE", Color::Orange)
      .value("WHITE", Color::White);

  py::enum_<ActionPrompt>(m, "ActionPrompt")
      .value("BUILD_INITIAL_SETTLEMENT", ActionPrompt::BuildInitialSettlement)
      .value("BUILD_INITIAL_ROAD", ActionPrompt::BuildInitialRoad)
      .value("PLAY_TURN", ActionPrompt::PlayTurn)
      .value("DISCARD", ActionPrompt::Discard)
      .value("MOVE_ROBBER", ActionPrompt::MoveRobber)
      .value("DECIDE_TRADE", ActionPrompt::DecideTrade)
      .value("DECIDE_ACCEPTEES", ActionPrompt::DecideAcceptees);

  py::enum_<ActionType>(m, "ActionType")
      .value("ROLL", ActionType::Roll)
      .value("MOVE_ROBBER", ActionType::MoveRobber)
      .value("DISCARD_RESOURCE", ActionType::DiscardResource)
      .value("BUILD_ROAD", ActionType::BuildRoad)
      .value("BUILD_SETTLEMENT", ActionType::BuildSettlement)
      .value("BUILD_CITY", ActionType::BuildCity)
      .value("BUY_DEVELOPMENT_CARD", ActionType::BuyDevelopmentCard)
      .value("PLAY_KNIGHT_CARD", ActionType::PlayKnightCard)
      .value("PLAY_YEAR_OF_PLENTY", ActionType::PlayYearOfPlenty)
      .value("PLAY_MONOPOLY", ActionType::PlayMonopoly)
      .value("PLAY_ROAD_BUILDING", ActionType::PlayRoadBuilding)
      .value("MARITIME_TRADE", ActionType::MaritimeTrade)
      .value("OFFER_TRADE", ActionType::OfferTrade)
      .value("ACCEPT_TRADE", ActionType::AcceptTrade)
      .value("REJECT_TRADE", ActionType::RejectTrade)
      .value("CONFIRM_TRADE", ActionType::ConfirmTrade)
      .value("CANCEL_TRADE", ActionType::CancelTrade)
      .value("END_TURN", ActionType::EndTurn);

  py::class_<Edge>(m, "Edge")
      .def(py::init([](int a, int b) { return Edge{a, b}; }))
      .def_readwrite("a", &Edge::a)
      .def_readwrite("b", &Edge::b)
      .def("normalized", &Edge::normalized)
      .def("__repr__", [](const Edge& edge) {
        Edge normalized = edge.normalized();
        return "Edge(" + std::to_string(normalized.a) + ", " + std::to_string(normalized.b) + ")";
      });

  py::class_<Action>(m, "Action")
      .def(py::init([](Color color, ActionType type, int value0, int value1) {
             return Action{color, type, value0, value1};
           }),
           py::arg("color"), py::arg("type"),
           py::arg("value0") = -1, py::arg("value1") = -1)
      .def_static("build_settlement", &Action::build_settlement)
      .def_static("build_road", &Action::build_road)
      .def_static("end_turn", &Action::end_turn)
      .def_readwrite("color", &Action::color)
      .def_readwrite("type", &Action::type)
      .def_readwrite("value0", &Action::value0)
      .def_readwrite("value1", &Action::value1);

  py::class_<ActionRecord>(m, "ActionRecord")
      .def_readwrite("action", &ActionRecord::action)
      .def_readwrite("result0", &ActionRecord::result0)
      .def_readwrite("result1", &ActionRecord::result1);

  py::class_<Player>(m, "Player")
      .def(py::init([](Color color, bool is_bot) { return Player{color, is_bot}; }),
           py::arg("color"), py::arg("is_bot") = true)
      .def_readwrite("color", &Player::color)
      .def_readwrite("is_bot", &Player::is_bot);

  py::class_<Game>(m, "Game")
      .def(py::init<std::vector<Player>, int>(), py::arg("players"), py::arg("vps_to_win") = 10)
      .def("execute", &Game::execute, py::arg("action"), py::arg("validate_action") = true)
      .def("generate_playable_actions", &Game::generate_playable_actions)
      .def("snapshot", [](const Game& game) { return snapshot_to_dict(game.snapshot()); });

  m.def("static_edges", []() {
    std::vector<std::pair<int, int>> result;
    for (const Edge& edge : static_edges()) {
      result.push_back({edge.a, edge.b});
    }
    return result;
  });
}
