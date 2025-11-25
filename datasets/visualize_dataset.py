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

    totals = df['total'].to_numpy()

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

    avg_scores = [df['objective'].mean(), df['cat'].mean(), df['color'].mean()]
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
    avg_total = df['total'].mean()
    print("\n--- Summary of Human Performance Data ---")
    print(f"Total Games Analyzed (n): {len(df)}")
    print(f"Overall Average Total Score: {avg_total:.2f}")
    print(f"Average Objective Score: {df['objective'].mean():.2f}")
    print(f"Average Cat Score: {df['cat'].mean():.2f}")
    print(f"Average Color Score: {df['color'].mean():.2f}")


if __name__ == '__main__':

    completed_data_df = pd.read_csv('human_dataset.csv')

    print("--- Analysis Starting ---")

    # Call the two separate functions
    generate_score_histogram(completed_data_df)
    generate_component_bar_chart(completed_data_df)