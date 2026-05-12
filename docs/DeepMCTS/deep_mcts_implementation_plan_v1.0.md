# Deep Monte Carlo Agent: Implementation Blueprint (Calico)

## 1. System Architecture
The agent transitions from a rollout-based MCTS to a Policy-Value guided search, decoupling game logic from neural perception.

### Component Overview
*   **CalicoEnv**: The pure state-machine (Rules, Legal Moves, Undo).
*   **CalicoEncoder**: Transforms board state into a $(C, H, W)$ tensor.
*   **DeepMCTSNode**: Stores Prior ($P$), Visits ($N$), and Value ($Q$).
*   **Neural Network**: A Dual-Head ResNet predicting Move Policy and Final Score.

---

## 2. Neural Network Design
The model uses a shared backbone to understand spatial board features, with two specialized output heads.

### Input Tensors
*   **Dimensions**: $(13, 7, 7)$
*   **Layers**: 6 Color planes, 6 Pattern planes, 1 Objective/Goal plane.

### Dual-Head Output
1.  **Policy Head**: Logits over all possible actions ($Softmax$).
2.  **Value Head**: Scalar representing expected final score (Linear/ReLU).

---

## 3. The Search Algorithm (PUCT)
Instead of random simulations, the agent navigates the tree using the **PUCT (Predictor + Upper Confidence Bound applied to Trees)** formula:

$$U(s, a) = Q(s, a) + C_{puct} \cdot P(s, a) \cdot \frac{\sqrt{N_{parent}}}{1 + N_{child}}$$

*   **Selection**: Follow the highest $U(s, a)$ until a leaf node is hit.
*   **Expansion**: Use the NN to generate $P$ values for all legal children at once.
*   **Evaluation**: Use the NN Value Head ($v$) to provide the reward signal.
*   **Backpropagation**: Update $Q$ and $N$ along the path.

---

## 4. The Training Pipeline (Self-Play)
Training is conducted in **Generations** to ensure data diversity and training stability.

### The Replay Buffer
*   **Role**: A "Memory Bank" that stores $(s, \pi, z)$ tuples.
*   **Benefit**: Breaks correlation between consecutive moves by sampling random mini-batches.

### The Trainer (Loss Functions)
The network optimizes a combined loss $L$:
1.  **Policy Loss**: Cross-Entropy between NN prediction and MCTS visit distribution.
2.  **Value Loss**: Mean Squared Error (MSE) between NN predicted score and actual game outcome.

### Generational Workflow
1.  **Self-Play Phase**: Play 20–50 games; store all moves in the Replay Buffer.
2.  **Training Phase**: Sample mini-batches from the buffer; perform 50–100 gradient descent steps.
3.  **Update**: Deploy the new model for the next generation of Self-Play.

---

## 5. Implementation Constraints
*   **No Comments**: Code must be clean and self-documenting via naming.
*   **Avoid Deepcopy**: Use `perform_action`/`undo_action` within the MCTS loop where possible to maximize iterations per second.
*   **Action Mapping**: A static mapper is required to link NN output indices to `CalicoAction` objects.