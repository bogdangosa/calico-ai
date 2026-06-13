import datetime
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalar_from_tfevents(log_dir, tag="Metrics/Average_Score"):
    """Extracts scalar data and timestamps from a single TensorBoard event file."""
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        return None
    
    # Take the latest event file in the directory
    event_file = max(event_files, key=os.path.getmtime)
    
    try:
        acc = EventAccumulator(event_file)
        acc.Reload()
        
        if tag not in acc.Tags()['scalars']:
            return None
            
        events = acc.Scalars(tag)
        # Store step, value, and wall_time (timestamp)
        return pd.DataFrame([(e.step, e.value, e.wall_time) for e in events], 
                            columns=['step', 'value', 'wall_time'])
    except Exception as e:
        print(f"Error reading {event_file}: {e}")
        return None

def format_duration(seconds):
    """Converts seconds to HH:MM:SS format."""
    return str(datetime.timedelta(seconds=int(seconds)))

def smooth_curve(points, factor=0.9):
    """Applies Exponential Moving Average smoothing."""
    smoothed = []
    if len(points) == 0:
        return smoothed
    last = points[0]
    for point in points:
        smoothed_val = last * factor + (1 - factor) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def main():
    # Configuration
    # Using absolute-style relative paths from project root
    log_root = "../../outputs/logs"
    
    # Define patterns to search for and their display names
    model_configs = [
        {"path": "tabular_td_training_v2/v1.0*", "label": "Rollout TD Agent"},
        {"path": "tabular_td_training_v2/simple*", "label": "Basic TD Agent"},
        {"path": "tabular_q_training_export/*", "label": "Q-Learning Agent"},
    ]
    
    output_plot = "../../outputs/plots/training_comparison2.pdf"
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Standard color palette
    
    found_data = False

    for i, config in enumerate(model_configs):
        search_path = os.path.join(log_root, config["path"])
        matching_dirs = [d for d in glob.glob(search_path) if os.path.isdir(d)]
        
        if not matching_dirs:
            continue
            
        # Use the most recent directory if multiple exist
        latest_dir = max(matching_dirs, key=os.path.getmtime)
        
        df = extract_scalar_from_tfevents(latest_dir)
        
        if df is not None and not df.empty:
            print(f"\n--- Model: {config['label']} ---")
            print(f"Directory: {latest_dir}")
            
            # Calculate Training Time
            start_time = df['wall_time'].min()
            end_time = df['wall_time'].max()
            duration_sec = end_time - start_time
            print(f"Training Duration: {format_duration(duration_sec)}")
            print(f"Total Episodes: {int(df['step'].max())}")
            print(f"Final Avg Score: {df['value'].iloc[-1]:.2f}")

            found_data = True
            # Plot raw data with low alpha (faded)
            plt.plot(df['step'], df['value'], alpha=0.15, color=sns_colors[i % len(sns_colors)])
            
            # Plot smoothed data
            smoothed_values = smooth_curve(df['value'].values, factor=0.88)
            plt.plot(df['step'], smoothed_values, label=config['label'], 
                     linewidth=2, color=sns_colors[i % len(sns_colors)])

    if found_data:
        plt.title("Comparison of Agent Training Performance (Micro Calico)", fontsize=14)
        plt.xlabel("Episodes", fontsize=12)
        plt.ylabel("Average Score", fontsize=12)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Save as PDF for paper quality (vector graphics)
        plt.savefig(output_plot, bbox_inches='tight')
        plt.savefig(output_plot.replace(".pdf", ".png"), bbox_inches='tight', dpi=300)
        print(f"\nPlot saved successfully to {output_plot} and .png version.")
    else:
        print("No data found to plot. Ensure training has started and event files are generated.")

if __name__ == "__main__":
    main()
