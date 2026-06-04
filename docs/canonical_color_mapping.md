# Canonical Color Mapping for State Space Reduction

## Goal
The primary objective of this algorithm is to compress the massive state space of the Calico board game by eliminating redundant color permutations. 

In Calico, a geometric pattern of tiles scores identical points regardless of the specific colors used, provided the structural relationships (e.g., matching groups, adjacencies) remain unchanged. For example, a cluster of three red tiles scores the exact same button points as a cluster of three blue tiles. Standard Q-learning treats these as entirely distinct states, forcing the network to waste training iterations learning the same foundational strategies for every color permutation. 

By mapping dynamic visual colors to a deterministic sequence of canonical IDs, we compress thousands of isomorphic board variations into a single unified representation, maximizing sample efficiency and accelerating network convergence.

---

## Algorithm Overview

The algorithm normalizes the environment data before it is encoded into a neural network tensor. It processes the game state in a strict, sequential order to build a dynamic translation dictionary.

### Steps
1. **Initialize Map:** Create an empty translation dictionary and set a canonical identifier counter to 0.
2. **Scan the Board:** Traverse the board grid sequentially (from top-left to bottom-right). For each tile encountered:
   - If the cell is empty, skip it.
   - If the tile color is already in the dictionary, replace its value with the mapped canonical ID.
   - If the tile color is new, add it to the dictionary, assign it the current counter value, increment the counter, and replace the tile value.
3. **Scan Secondary Features:** Process the player's hand, available market tiles, and visible public objectives using the *same* dictionary built during the board scan. If a new color appears in these steps that wasn't on the board, continue appending it to the dictionary using the sequential counter.
4. **Output:** Return the structurally identical but color-neutralized board state components for tensor encoding.

---

## Mathematical Impact

Calico features 6 distinct tile colors. Without reduction, a single board configuration can manifest in any of the color permutations:

$$\text{Permutations} = 6! = 720$$

By mapping colors strictly by order of appearance, all 720 variations collapse down into **1 unique canonical state**. A network trained on this representation generalizes an experienced layout instantly to all other equivalent color schemes without ever having seen them during training.

---

## Reference Implementation