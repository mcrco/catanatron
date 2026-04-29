import { useEffect, useRef, useState, useContext } from "react";
import { useParams } from "react-router-dom";
import PropTypes from "prop-types";
import { GridLoader } from "react-spinners";
import { useSnackbar } from "notistack";

import ZoomableBoard from "./ZoomableBoard";
import ActionsToolbar from "./ActionsToolbar";

import "./GameScreen.scss";
import LeftDrawer from "../components/LeftDrawer";
import RightDrawer from "../components/RightDrawer";
import { store } from "../store";
import ACTIONS from "../actions";
import { type StateIndex, getState, postAction } from "../utils/apiClient";
import { dispatchSnackbar } from "../components/Snackbar";
import { getHumanColor } from "../utils/stateUtils";
import AnalysisBox from "../components/AnalysisBox";
import { Divider } from "@mui/material";

const ROBOT_THINKING_TIME = 300;

function GameScreen({ replayMode }: { replayMode: boolean }) {
  const { gameId, stateIndex } = useParams();
  const { state, dispatch } = useContext(store);
  const { enqueueSnackbar, closeSnackbar } = useSnackbar();
  const [isBotThinking, setIsBotThinking] = useState(false);
  const botRequestKeyRef = useRef<string | null>(null);

  // Load game state
  useEffect(() => {
    if (!gameId) {
      return;
    }

    (async () => {
      const gameState = await getState(gameId, stateIndex as StateIndex);
      dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
    })();
  }, [gameId, stateIndex, dispatch]);

  // Maybe kick off next query?
  useEffect(() => {
    if (!state.gameState || replayMode || !gameId) {
      return;
    }
    if (
      state.gameState.bot_colors.includes(state.gameState.current_color) &&
      !state.gameState.winning_color
    ) {
      const requestKey = [
        gameId,
        state.gameState.state_index,
        state.gameState.current_color,
        state.gameState.current_prompt,
      ].join(":");
      if (botRequestKeyRef.current === requestKey) {
        return;
      }
      botRequestKeyRef.current = requestKey;

      // Make bot click next action.
      (async () => {
        setIsBotThinking(true);
        const start = new Date();
        try {
          const gameState = await postAction(gameId);
          const requestTime = new Date().valueOf() - start.valueOf();
          setTimeout(() => {
            // simulate thinking
            setIsBotThinking(false);
            dispatch({ type: ACTIONS.SET_GAME_STATE, data: gameState });
            if (getHumanColor(gameState)) {
              dispatchSnackbar(enqueueSnackbar, closeSnackbar, gameState);
            }
          }, Math.max(0, ROBOT_THINKING_TIME - requestTime));
        } catch (error) {
          botRequestKeyRef.current = null;
          setIsBotThinking(false);
          throw error;
        }
      })();
    }
  }, [
    gameId,
    replayMode,
    state.gameState,
    dispatch,
    enqueueSnackbar,
    closeSnackbar,
  ]);

  if (!state.gameState) {
    return (
      <main>
        <GridLoader
          className="loader"
          color="#000000"
          size={100}
        />
      </main>
    );
  }

  return (
    <main>
      <h1 className="logo">Catanatron</h1>
      <ZoomableBoard replayMode={replayMode} />
      <ActionsToolbar isBotThinking={isBotThinking} replayMode={replayMode} />
      <LeftDrawer />
      <RightDrawer>
        <AnalysisBox stateIndex={"latest"}/>
        <Divider />
      </RightDrawer>
    </main>
  );
}

GameScreen.propTypes = {
  /**
   * Injected by the documentation to work in an iframe.
   * You won't need it on your project.
   */
  window: PropTypes.func,
};

export default GameScreen;
