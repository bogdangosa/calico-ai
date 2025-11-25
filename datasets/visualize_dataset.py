import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def generate_score_histogram(df: pd.DataFrame):
    """
    Generates and displays a histogram of the total scores.
    """
    if df.empty:
        print("No data to analyze.")
        return

    totals = df['score'].to_numpy()

    plt.figure(figsize=(8, 6))
    plt.hist(totals, bins=np.arange(min(totals) - 1, max(totals) + 3, 3),
             edgecolor='black', alpha=0.7, color='teal')

    avg_total = np.mean(totals)
    plt.axvline(avg_total, color='red', linestyle='dashed', linewidth=1.5,
                label=f'Average Score: {avg_total:.2f}')

    plt.title('Distribution of Human Player Total Scores (n=' + str(len(totals)) + ')')
    plt.xlabel('Total Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()


def generate_component_bar_chart(df: pd.DataFrame):
    """
    Generates and displays a bar chart of average component scores.
    """
    if df.empty:
        print("No data to analyze.")
        return

    avg_scores = [df['objectives_score'].mean(), df['cats_score'].mean(), df['color_score'].mean()]
    component_labels = ['Objective Potential', 'Cat Potential', 'Color Potential']

    plt.figure(figsize=(7, 6))
    bars = plt.bar(component_labels, avg_scores, color=['orange', 'red', 'green'], alpha=0.8)

    plt.title('Average Contribution of Scoring Components')
    plt.ylabel('Average Points Awarded')
    plt.grid(axis='y', linestyle='--')

    # Add labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f'{yval:.2f}', ha='center', fontsize=12)

    plt.tight_layout()
    plt.show()

    # Console output summary (Consolidated here for simplicity in execution)
    avg_total = df['score'].mean()
    print("\n--- Summary of Human Performance Data ---")
    print(f"Total Games Analyzed (n): {len(df)}")
    print(f"Overall Average Total Score: {avg_total:.2f}")
    print(f"Average Objective Score: {df['objectives_score'].mean():.2f}")
    print(f"Average Cat Score: {df['cats_score'].mean():.2f}")
    print(f"Average Color Score: {df['color_score'].mean():.2f}")


def compare_multiple_runs(file_paths: list[str]):
    """
    Loads data from multiple CSV files, aggregates all score components (Total, Objective, Cat, Color)
    for each, and displays a grouped bar chart comparing their performance profiles.

    The files are expected to have 'score', 'objectives_score', 'cats_score', and 'color_score' columns.
    """
    all_data = []

    print(f"--- Loading and Comparing {len(file_paths)} Runs ---")

    for file_path in file_paths:
        try:
            # Extract a meaningful name (e.g., filename without extension)
            run_name = os.path.basename(file_path).replace('.csv', '')

            # Read the CSV file
            df = pd.read_csv(file_path)

            required_cols = ['score', 'objectives_score', 'cats_score', 'color_score']
            if not all(col in df.columns for col in required_cols):
                print(f"Error: File {file_path} missing one or more required score columns. Skipping.")
                continue

            # Calculate means
            means = df[required_cols].mean().to_dict()

            # Store the aggregate data
            all_data.append({
                'Run': run_name,
                'Total': means['score'],
                'Objective': means['objectives_score'],
                'Cat': means['cats_score'],
                'Color': means['color_score']
            })
            print(f"Loaded {run_name}: Average Total Score = {means['score']:.2f}")

        except FileNotFoundError:
            print(f"Error: File not found at {file_path}. Skipping.")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}. Skipping.")

    if not all_data:
        print("No valid data loaded for comparison.")
        return

    # Convert aggregate list to DataFrame for easier plotting
    comparison_df = pd.DataFrame(all_data).set_index('Run')

    # Define plotting structure
    score_components = ['Objective', 'Cat', 'Color', 'Total']
    colors = ['orange', 'red', 'green', 'blue']

    # We will plot the three components and Total separately, stacking them slightly differently

    fig, ax = plt.subplots(figsize=(12, 7))

    # --- Plotting Grouped Bar Chart of Component Scores ---

    # Select only Objective, Cat, and Color for the grouped bar chart
    components_df = comparison_df[['Objective', 'Cat', 'Color']]

    # Plot the components
    components_df.plot(kind='bar', ax=ax, color=colors[:3], alpha=0.8, width=0.8)

    # Calculate bar positions for placing total score text
    bar_width = 0.8 / components_df.shape[1]
    x_positions = np.arange(len(components_df.index))

    # Add Total Score text above the grouped bars
    for i, run_name in enumerate(components_df.index):
        total_score = comparison_df.loc[run_name, 'Total']

        # Calculate the center position of the group of bars
        center_x = x_positions[i]

        ax.text(center_x, total_score + 1.5, f'Total: {total_score:.2f}',
                ha='center', fontsize=10, weight='bold', color='black')

    ax.set_title('Comparison of Average Score Breakdown Across Experimental Runs', fontsize=14)
    ax.set_ylabel('Average Points Awarded', fontsize=12)
    ax.set_xlabel('Experiment / Agent Configuration', fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(components_df.index, rotation=0)
    ax.legend(title='Score Type', loc='upper left')
    ax.grid(axis='y', linestyle='--')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # --- Example Usage for compare_multiple_runs ---

    # NOTE: You must replace these paths with the actual paths to your logged CSV files!
    example_files = [
        'agent_lookahead_results.csv',
        'agent_lookahead_results_optimised.csv',
        'human_dataset.csv'
    ]

    compare_multiple_runs(example_files)