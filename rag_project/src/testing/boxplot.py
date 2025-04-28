import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

sns.set_theme(style="whitegrid", palette="Set2")

def main():
    metrics_data = {}
    results_dir = os.path.join(os.getcwd(), 'test_results')
    for filepath in glob.glob(os.path.join(results_dir, '*.json')):
        with open(filepath) as f:
            data = json.load(f)
        metrics_scores = {}
        for entry in data:
            for metric, details in entry['metrics'].items():
                metrics_scores.setdefault(metric, []).append(details['score'])
        basename = os.path.basename(filepath)
        match = re.search(r'iter_(\d+)_', basename)
        if not match:
            print(f"Skipping {basename}: no iteration number found.")
            continue
        iteration = int(match.group(1))
        for metric, scores in metrics_scores.items():
            metrics_data.setdefault(metric, {}).setdefault(iteration, []).extend(scores)
        means = {metric: pd.Series(scores).mean() for metric, scores in metrics_scores.items()}
        std_devs = {metric: pd.Series(scores).std() for metric, scores in metrics_scores.items()}
        print(f"Mean and Standard Deviation of Metrics in {os.path.basename(filepath)}:")
        df = pd.DataFrame({'mean': pd.Series(means), 'std_dev': pd.Series(std_devs)})
        print(df)

        metrics_df = pd.DataFrame(metrics_scores)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=metrics_df)
        plt.title(f"Boxplot of Metrics in {os.path.basename(filepath)}")
        plt.ylabel("Score")
        plt.xlabel("Metric")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_dir = os.path.join(results_dir, 'boxplots')
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{os.path.splitext(os.path.basename(filepath))[0]}_boxplot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved boxplot to {plot_path}")

    print("Generating aggregated boxplots per metric across iterations...")
    for metric, iter_scores in metrics_data.items():
        # Build DataFrame allowing uneven lengths by concatenating Series
        series_list = [pd.Series(iter_scores[it], name=str(it)) for it in sorted(iter_scores)]
        agg_df = pd.concat(series_list, axis=1)
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=agg_df)
        plt.title(f"Aggregated Boxplot for {metric} across iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        agg_plot_path = os.path.join(results_dir, 'boxplots', f"{metric}_aggregated_boxplot.png")
        plt.savefig(agg_plot_path)
        plt.close()
        print(f"Saved aggregated boxplot to {agg_plot_path}")

if __name__ == "__main__":
    main()
