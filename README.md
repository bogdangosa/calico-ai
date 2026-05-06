# Calico AI

Calico AI is a comprehensive simulation and artificial intelligence framework for the board game **Calico**. It provides a robust environment for game simulation, a suite of AI agents ranging from simple heuristics to advanced tree searches, and tools for benchmarking and data collection.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Console Game](#console-game)
  - [AI Agents](#ai-agents)
  - [FastAPI Service](#fastapi-service)
- [AI Agents](#ai-agents-detail)
- [Configuration](#configuration)
- [Research and Experiments](#research-and-experiments)

## Overview

Calico is a puzzle-y tile-laying game where players compete to sew the coziest quilt as they collect and place patches of different colors and patterns. This project implements the game logic (engine), scoring mechanisms, and various AI strategies to play the game optimally.

## Key Features

- **Game Engine**: A high-fidelity implementation of Calico rules, including tile placement, shop management, and board constraints.
- **Scoring System**: Detailed scoring logic for color groups, pattern-based cat goals, and board objectives.
- **Multiple AI Agents**:
  - **Lookahead Agents**: 1-step, Multi-step, and Top-K lookahead strategies.
  - **Monte Carlo Agents**: Flat Monte Carlo and Monte Carlo Tree Search (MCTS).
  - **Random Agent**: Baseline for benchmarking.
- **Extensible Configuration**: JSON-based settings for different game variants (Micro, Mini, Full).
- **FastAPI Integration**: Serve AI moves via a REST API.
- **Data Collection**: Scripts for simulating games and saving results for further analysis.

## Project Structure

```text
├── agents/             # Pre-trained models and weights for AI agents
├── config/             # Game settings and configurations (Micro, Mini, Full Calico)
├── datasets/           # Collected simulation data and datasets
├── experiments/        # Research scripts for TD-Learning, DQN, and hybrid solutions
├── src/
│   ├── agents/         # AI agent implementations (Lookahead, Monte Carlo, TD)
│   ├── engine/         # Core game logic: environment, scoring, and history
│   ├── models/         # Pydantic models for game configuration and state
│   ├── scripts/        # Entry points for running games and simulations
│   ├── ui/             # Console-based UI and rendering tools
│   └── utils/          # Configuration loading, timing, and visualization utilities
├── main.py             # FastAPI entry point
└── tests/              # Project test suite
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/calico-ai.git
   cd calico-ai
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   *(Note: Ensure you have `numpy`, `pydantic`, `fastapi`, and `uvicorn` installed.)*

## Usage

### Console Game
Play the game or watch an AI play in the console:
```bash
python src/scripts/run_console_game.py
```

### AI Agents
Run specific agent simulations:
```bash
python src/scripts/run_lookahead_agent.py
python src/scripts/run_monte_carlo_treesearch_agent.py
```

### FastAPI Service
Start the API server to get AI moves via HTTP:
```bash
uvicorn main:app --reload
```

## AI Agents Detail

- **Lookahead Agents**:
  - `OneStepLookaheadAgent`: Evaluates all immediate legal moves and picks the one with the highest expected score.
  - `MultiStepLookaheadAgent`: Explores the game tree to a specified depth.
  - `TopKLookaheadAgent`: A more efficient lookahead that only explores the top-K promising moves at each level.
- **Monte Carlo Agents**:
  - `FlatMonteCarloAgent`: Plays many random games from the current state and picks the move with the best average outcome.
  - `MonteCarloTreeSearchAgent`: Uses MCTS (Selection, Expansion, Simulation, Backpropagation) to find optimal moves.
- **Temporal Difference (TD) Learning**: Located in `experiments/`, these agents use reinforcement learning to learn a value function for game states.

## Configuration

The game behavior is controlled via JSON files in the `config/` directory. You can adjust:
- `board.size`: Dimension of the quilt.
- `tiles`: Number of colors, patterns, and identical tiles.
- `evaluation`: Weights for different scoring components (color potential, cat potential, etc.).

Example configuration (`micro_calico_settings.json`):
```json
{
    "name": "micro_calico",
    "player_hand_size": 2,
    "nr_of_tiles_in_shop": 3,
    "board": {
        "size": 4,
        "no_tile_value": 37
    }
}
```

## Research and Experiments

The `experiments/` folder contains ongoing research into more advanced AI techniques:
- `Deep Q-Learning (DQN)`: Training neural networks to play Calico.
- `Value Estimators`: Comparing different ways to estimate the potential of a partial board.
- `Hybrid Solutions`: Combining lookahead with learned value functions.
