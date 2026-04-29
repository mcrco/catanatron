import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Alert,
  Button,
  Checkbox,
  IconButton,
  MenuItem,
  Select,
  Slider,
  Tooltip,
} from "@mui/material";
import HelpOutlineRoundedIcon from "@mui/icons-material/HelpOutlineRounded";
import { GridLoader } from "react-spinners";
import {
  createGame,
  getPlayers,
  type MapTemplate,
  type PlayerOption,
  type PlayerArchetype,
} from "../utils/apiClient";

import "./HomePage.scss";

const FALLBACK_PLAYER_OPTIONS: PlayerOption[] = [
  {
    key: "HUMAN",
    label: "Human",
    description: "Human player controlled from the web UI.",
    min_players: 2,
    max_players: 4,
    map_templates: ["BASE", "MINI", "TOURNAMENT"],
  },
  {
    key: "RANDOM",
    label: "Random",
    description: "Chooses legal actions uniformly at random.",
    min_players: 2,
    max_players: 4,
    map_templates: ["BASE", "MINI", "TOURNAMENT"],
  },
  {
    key: "CATANATRON",
    label: "Catanatron",
    description: "Built-in alpha-beta Catanatron bot.",
    min_players: 2,
    max_players: 4,
    map_templates: ["BASE", "MINI", "TOURNAMENT"],
  },
  {
    key: "WEIGHTED_RANDOM",
    label: "Weighted Random",
    description: "Random player biased toward high-value build actions.",
    min_players: 2,
    max_players: 4,
    map_templates: ["BASE", "MINI", "TOURNAMENT"],
  },
];

const MAP_TEMPLATES: MapTemplate[] = ["BASE", "MINI", "TOURNAMENT"];
const PLAYER_COLORS = ["RED", "BLUE", "ORANGE", "WHITE"] as const;

function isOptionUnsupported(
  option: PlayerOption,
  playerCount: number,
  mapTemplate: MapTemplate
) {
  return (
    playerCount < option.min_players ||
    playerCount > option.max_players ||
    !option.map_templates.includes(mapTemplate)
  );
}

export default function HomePage() {
  const [loading, setLoading] = useState(false);
  const [mapTemplate, setMapTemplate] = useState<MapTemplate>("BASE");
  const [vpsToWin, setVpsToWin] = useState(15);
  const [discardLimit, setDiscardLimit] = useState(9);
  const [friendlyRobber, setFriendlyRobber] = useState(true);
  const [playerOptions, setPlayerOptions] = useState<PlayerOption[]>(
    FALLBACK_PLAYER_OPTIONS
  );
  const [players, setPlayers] = useState<PlayerArchetype[]>([
    "HUMAN",
    "CATANATRON",
  ]);
  const navigate = useNavigate();
  const humanCount = players.filter((player) => player === "HUMAN").length;
  const hasTooManyHumans = humanCount > 1;
  const hasUnsupportedPlayers = players.some((player) => {
    const option = playerOptions.find((candidate) => candidate.key === player);
    return option ? isOptionUnsupported(option, players.length, mapTemplate) : false;
  });

  useEffect(() => {
    getPlayers()
      .then((options) => {
        if (options.length > 0) {
          setPlayerOptions(options);
        }
      })
      .catch((error) => {
        console.error("Failed to fetch player options:", error);
      });
  }, []);

  const handlePlayerChange = (index: number, value: PlayerArchetype) => {
    if (
      value === "HUMAN" &&
      players[index] !== "HUMAN" &&
      humanCount >= 1
    ) {
      return;
    }

    setPlayers((current) =>
      current.map((player, playerIndex) =>
        playerIndex === index ? value : player
      )
    );
  };

  const handleAddPlayer = () => {
    const defaultBot =
      playerOptions.find(
        (option) =>
          option.key !== "HUMAN" &&
          !isOptionUnsupported(option, players.length + 1, mapTemplate)
      )?.key ?? "WEIGHTED_RANDOM";
    setPlayers((current) =>
      current.length >= 4 ? current : [...current, defaultBot]
    );
  };

  const handleRemovePlayer = (index: number) => {
    setPlayers((current) =>
      current.length <= 2
        ? current
        : current.filter((_, playerIndex) => playerIndex !== index)
    );
  };

  const handleCreateGame = async () => {
    if (hasTooManyHumans) {
      return;
    }

    setLoading(true);
    const gameId = await createGame({
      players,
      mapTemplate,
      vpsToWin,
      discardLimit,
      friendlyRobber,
    });
    setLoading(false);
    navigate("/games/" + gameId);
  };

  return (
    <div className="home-page">
      <div className="home-page__inner">
        <h1 className="logo">Catanatron</h1>
        {!loading ? (
          <div className="setup-card">
            <p className="setup-note">Open hands. Random discard choice.</p>

            <div className="control-group">
              <div className="control-header">
                <span>Map Template</span>
                <strong>{mapTemplate}</strong>
              </div>
              <div className="map-template-buttons">
                {MAP_TEMPLATES.map((value) => (
                  <Button
                    key={value}
                    variant="contained"
                    onClick={() => setMapTemplate(value)}
                    className={`choice-button ${
                      mapTemplate === value ? "selected" : ""
                    }`}
                  >
                    {value}
                  </Button>
                ))}
              </div>
            </div>

            <div className="control-row">
              <div className="control-group compact-control">
                <div className="control-header">
                  <span>Points to Win</span>
                  <strong>{vpsToWin}</strong>
                </div>
                <Slider
                  value={vpsToWin}
                  min={3}
                  max={20}
                  step={1}
                  marks
                  valueLabelDisplay="auto"
                  onChange={(_, value) => setVpsToWin(value as number)}
                />
              </div>

              <div className="control-group compact-control">
                <div className="control-header">
                  <span>Card Discard Limit</span>
                  <strong>{discardLimit}</strong>
                </div>
                <Slider
                  value={discardLimit}
                  min={5}
                  max={20}
                  step={1}
                  marks
                  valueLabelDisplay="auto"
                  onChange={(_, value) => setDiscardLimit(value as number)}
                />
              </div>

              <div className="control-group compact-control switch-control">
                <div className="control-header">
                  <span className="inline-title">
                    Friendly Robber
                    <Tooltip
                      title="Prevent robber placement on tiles touching opponents with 2 victory points."
                      arrow
                      enterTouchDelay={0}
                      leaveTouchDelay={3000}
                    >
                      <IconButton
                        size="small"
                        className="help-button"
                        aria-label="Friendly Robber help"
                      >
                        <HelpOutlineRoundedIcon fontSize="inherit" />
                      </IconButton>
                    </Tooltip>
                  </span>
                  <strong>{friendlyRobber ? "On" : "Off"}</strong>
                </div>
                <Checkbox
                  className="inline-switch"
                  checked={friendlyRobber}
                  onChange={(event) =>
                    setFriendlyRobber(event.target.checked)
                  }
                />
              </div>
            </div>

            <div className="control-group">
              <div className="control-header">
                <span>Players</span>
                <strong>{players.length}/4</strong>
              </div>
              {hasTooManyHumans && (
                <Alert severity="error" className="players-alert">
                  Only one Human player is allowed.
                </Alert>
              )}
              {hasUnsupportedPlayers && (
                <Alert severity="error" className="players-alert">
                  One or more selected players do not support this map or
                  player count.
                </Alert>
              )}
              <div className="players-list">
                {players.map((player, index) => (
                  <div className="player-row" key={`${player}-${index}`}>
                    <div className="player-meta">
                      <span className="player-label">Player {index + 1}</span>
                      <span
                        className={`player-color-chip ${PLAYER_COLORS[index].toLowerCase()}`}
                      >
                        {PLAYER_COLORS[index]}
                      </span>
                    </div>
                    <Select
                      size="small"
                      fullWidth
                      value={player}
                      onChange={(event) =>
                        handlePlayerChange(
                          index,
                          event.target.value as PlayerArchetype
                        )
                      }
                    >
                      {playerOptions.map((option) => (
                        <MenuItem
                          key={option.key}
                          value={option.key}
                          disabled={
                            (option.key === "HUMAN" &&
                              humanCount >= 1 &&
                              player !== "HUMAN") ||
                            isOptionUnsupported(
                              option,
                              players.length,
                              mapTemplate
                            )
                          }
                        >
                          {option.label}
                        </MenuItem>
                      ))}
                    </Select>
                    <Button
                      variant="text"
                      className="remove-player-btn"
                      disabled={players.length <= 2}
                      onClick={() => handleRemovePlayer(index)}
                    >
                      Remove
                    </Button>
                  </div>
                ))}
              </div>

              <Button
                variant="contained"
                className="add-player-btn"
                disabled={players.length >= 4}
                onClick={handleAddPlayer}
              >
                Add Player
              </Button>
            </div>

            <Button
              variant="contained"
              color="primary"
              className="start-btn"
              disabled={hasTooManyHumans || hasUnsupportedPlayers}
              onClick={handleCreateGame}
            >
              Start
            </Button>
          </div>
        ) : (
          <GridLoader
            className="loader"
            color="#ffffff"
            size={60}
          />
        )}
        <div className="page-end-spacer" aria-hidden="true" />
      </div>
    </div>
  );
}
