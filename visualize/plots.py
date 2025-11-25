import matplotlib.pyplot as plt
import numpy as np

def plot_score_distribution(scores):
    """
    Plots a bar chart showing the distribution of scores.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=15, color='skyblue', edgecolor='black')
    plt.title("Distribution of Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_average_convergence(average_scores):
    """
    Plots a line chart showing how the average score converges over time.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(average_scores)+1), average_scores, marker='o', color='orange')
    plt.title("Average Score Convergence")
    plt.xlabel("Number of Games")
    plt.ylabel("Average Score")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
