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


import pandas as pd
import matplotlib.pyplot as plt


def plot_average_scores_excluding_total(file_path):
    experiment_data = pd.read_csv(file_path)

    target_columns = ['color_score', 'objective_score', 'cat_score']
    average_scores = experiment_data[target_columns].mean()

    plt.figure(figsize=(8, 6))
    bar_chart = plt.bar(average_scores.index, average_scores.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

    plt.title('Average Score Across Experiments')
    plt.ylabel('Average Score')
    plt.xlabel('Score Category')

    for bar in bar_chart:
        bar_height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, bar_height + 0.1,
                 f'{bar_height:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_average_scores_excluding_total('../datasets/main_calico/SB3Agent/simulation_m2f1tkvd.csv')