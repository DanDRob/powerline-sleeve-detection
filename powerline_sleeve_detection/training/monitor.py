import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """Monitor and visualize training progress for powerline sleeve detection models."""

    def __init__(self, output_dir: Union[str, Path]):
        """Initialize the training monitor.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def load_results(self, results_path: Union[str, Path]) -> pd.DataFrame:
        """Load training results from a results.csv file.

        Args:
            results_path: Path to results.csv file

        Returns:
            DataFrame with training metrics
        """
        results_path = Path(results_path)
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")

        try:
            # Load the CSV file
            df = pd.read_csv(results_path)
            logger.info(f"Loaded training results from {results_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load results from {results_path}: {e}")
            raise

    def plot_metrics(self,
                     results_df: pd.DataFrame,
                     metrics: Optional[List[str]] = None,
                     include_validation: bool = True,
                     smooth: bool = True,
                     smooth_window: int = 5) -> Dict[str, Path]:
        """Plot training metrics over epochs.

        Args:
            results_df: DataFrame with training results
            metrics: List of metrics to plot (if None, plots all metrics)
            include_validation: Whether to include validation metrics
            smooth: Whether to apply smoothing to the plots
            smooth_window: Window size for smoothing

        Returns:
            Dictionary mapping metric names to plot file paths
        """
        # If metrics not specified, use all metrics except epoch
        if metrics is None:
            metrics = [col for col in results_df.columns if col !=
                       'epoch' and not col.startswith('val_')]

        plot_paths = {}

        # Create a figure for each metric
        for metric in metrics:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))

                # Get validation metric name if available
                val_metric = f"val_{metric}" if include_validation else None

                # Apply smoothing if requested
                if smooth and len(results_df) > smooth_window:
                    # Create a smoothed series
                    smoothed = results_df[metric].rolling(
                        window=smooth_window, center=True).mean()
                    ax.plot(results_df['epoch'], results_df[metric],
                            'b-', alpha=0.3, label=f"{metric} (raw)")
                    ax.plot(results_df['epoch'], smoothed, 'b-',
                            linewidth=2, label=f"{metric} (smoothed)")

                    if val_metric and val_metric in results_df.columns:
                        val_smoothed = results_df[val_metric].rolling(
                            window=smooth_window, center=True).mean()
                        ax.plot(results_df['epoch'], results_df[val_metric],
                                'r-', alpha=0.3, label=f"{val_metric} (raw)")
                        ax.plot(results_df['epoch'], val_smoothed, 'r-',
                                linewidth=2, label=f"{val_metric} (smoothed)")
                else:
                    # Plot without smoothing
                    ax.plot(results_df['epoch'], results_df[metric],
                            'b-', linewidth=2, label=metric)

                    if val_metric and val_metric in results_df.columns:
                        ax.plot(
                            results_df['epoch'], results_df[val_metric], 'r-', linewidth=2, label=val_metric)

                # Set labels and title
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.set_title(f'Training Progress - {metric}')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()

                # Save the figure
                output_path = self.output_dir / f"{metric}_plot.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

                plot_paths[metric] = output_path
                logger.info(f"Created plot for {metric} at {output_path}")

            except Exception as e:
                logger.error(f"Error creating plot for {metric}: {e}")

        return plot_paths

    def create_training_summary(self,
                                results_df: pd.DataFrame,
                                model_info: Optional[Dict[str, Any]] = None) -> Path:
        """Create a comprehensive training summary report.

        Args:
            results_df: DataFrame with training results
            model_info: Optional dictionary with model information

        Returns:
            Path to the generated summary file
        """
        summary_path = self.output_dir / "training_summary.html"

        try:
            # Extract key metrics
            last_epoch = results_df['epoch'].max()

            # Get final values for key metrics
            final_metrics = {}
            val_metrics = {}

            for col in results_df.columns:
                if col != 'epoch':
                    if col.startswith('val_'):
                        val_metrics[col] = results_df[col].iloc[-1]
                    else:
                        final_metrics[col] = results_df[col].iloc[-1]

            # Create HTML content
            html_content = [
                "<!DOCTYPE html>",
                "<html>",
                "<head>",
                "    <title>Training Summary Report</title>",
                "    <style>",
                "        body { font-family: Arial, sans-serif; margin: 20px; }",
                "        .container { max-width: 1200px; margin: 0 auto; }",
                "        .header { background-color: #f1f1f1; padding: 20px; border-radius: 5px; }",
                "        .section { margin-top: 30px; }",
                "        table { border-collapse: collapse; width: 100%; }",
                "        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }",
                "        th { background-color: #f2f2f2; }",
                "        .plot { margin: 20px 0; }",
                "    </style>",
                "</head>",
                "<body>",
                "    <div class='container'>",
                "        <div class='header'>",
                f"            <h1>Training Summary Report</h1>",
                f"            <p>Total Epochs: {last_epoch}</p>",
                "        </div>"
            ]

            # Add model info if provided
            if model_info:
                html_content.extend([
                    "        <div class='section'>",
                    "            <h2>Model Information</h2>",
                    "            <table>",
                    "                <tr><th>Parameter</th><th>Value</th></tr>"
                ])

                for key, value in model_info.items():
                    html_content.append(
                        f"                <tr><td>{key}</td><td>{value}</td></tr>")

                html_content.append("            </table>")
                html_content.append("        </div>")

            # Add final metrics section
            html_content.extend([
                "        <div class='section'>",
                "            <h2>Final Training Metrics</h2>",
                "            <table>",
                "                <tr><th>Metric</th><th>Value</th></tr>"
            ])

            for metric, value in final_metrics.items():
                html_content.append(
                    f"                <tr><td>{metric}</td><td>{value:.6f}</td></tr>")

            html_content.append("            </table>")
            html_content.append("        </div>")

            # Add validation metrics if available
            if val_metrics:
                html_content.extend([
                    "        <div class='section'>",
                    "            <h2>Final Validation Metrics</h2>",
                    "            <table>",
                    "                <tr><th>Metric</th><th>Value</th></tr>"
                ])

                for metric, value in val_metrics.items():
                    # Remove 'val_' prefix for display
                    display_name = metric[4:] if metric.startswith(
                        'val_') else metric
                    html_content.append(
                        f"                <tr><td>{display_name}</td><td>{value:.6f}</td></tr>")

                html_content.append("            </table>")
                html_content.append("        </div>")

            # Add plots section
            metrics_to_plot = [
                col for col in results_df.columns if col != 'epoch' and not col.startswith('val_')]

            html_content.extend([
                "        <div class='section'>",
                "            <h2>Training Progress</h2>"
            ])

            # Create and embed plots
            for metric in metrics_to_plot:
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))

                # Get validation metric name if available
                val_metric = f"val_{metric}"

                # Plot training metric
                ax.plot(results_df['epoch'], results_df[metric],
                        'b-', linewidth=2, label=metric)

                # Plot validation metric if available
                if val_metric in results_df.columns:
                    ax.plot(results_df['epoch'], results_df[val_metric],
                            'r-', linewidth=2, label=val_metric)

                # Set labels and title
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.set_title(f'Training Progress - {metric}')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()

                # Save the figure
                plot_path = self.output_dir / f"{metric}_plot.png"
                fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

                # Add to HTML
                html_content.append(f"            <div class='plot'>")
                html_content.append(f"                <h3>{metric}</h3>")
                html_content.append(
                    f"                <img src='{plot_path.name}' alt='{metric} plot' style='max-width:100%;'>")
                html_content.append(f"            </div>")

            # Close HTML tags
            html_content.extend([
                "        </div>",
                "    </div>",
                "</body>",
                "</html>"
            ])

            # Write HTML file
            with open(summary_path, 'w') as f:
                f.write('\n'.join(html_content))

            logger.info(f"Created training summary at {summary_path}")
            return summary_path

        except Exception as e:
            logger.error(f"Error creating training summary: {e}")
            raise

    def plot_confusion_matrix(self,
                              matrix: np.ndarray,
                              class_names: List[str],
                              title: str = "Confusion Matrix") -> Path:
        """Plot confusion matrix.

        Args:
            matrix: Confusion matrix as numpy array
            class_names: List of class names
            title: Plot title

        Returns:
            Path to the saved plot
        """
        try:
            plt.figure(figsize=(10, 8))

            # Plot with seaborn for better visualization
            sns.heatmap(
                matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names
            )

            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(title)

            # Save the figure
            output_path = self.output_dir / "confusion_matrix.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Created confusion matrix plot at {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error creating confusion matrix plot: {e}")
            raise

    def compare_models(self,
                       results_paths: List[Union[str, Path]],
                       model_names: List[str],
                       metrics: Optional[List[str]] = None) -> Dict[str, Path]:
        """Compare multiple models by plotting their metrics together.

        Args:
            results_paths: List of paths to results.csv files
            model_names: List of model names (labels for the plot)
            metrics: List of metrics to compare (if None, uses common metrics)

        Returns:
            Dictionary mapping metric names to plot file paths
        """
        if len(results_paths) != len(model_names):
            raise ValueError(
                "Number of results paths must match number of model names")

        # Load all results into dataframes
        dfs = []
        for path in results_paths:
            try:
                df = self.load_results(path)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading results from {path}: {e}")
                return {}

        # Find common metrics if not specified
        if metrics is None:
            # Get common metrics across all dataframes
            common_cols = set(dfs[0].columns)
            for df in dfs[1:]:
                common_cols &= set(df.columns)

            metrics = [col for col in common_cols if col !=
                       'epoch' and not col.startswith('val_')]

        plot_paths = {}

        # Create comparison plots for each metric
        for metric in metrics:
            try:
                fig, ax = plt.subplots(figsize=(12, 8))

                for i, (df, name) in enumerate(zip(dfs, model_names)):
                    if metric in df.columns:
                        ax.plot(df['epoch'], df[metric], linewidth=2,
                                label=f"{name} - {metric}")

                    # Add validation metric if available
                    val_metric = f"val_{metric}"
                    if val_metric in df.columns:
                        ax.plot(df['epoch'], df[val_metric], '--',
                                linewidth=2, label=f"{name} - {val_metric}")

                # Set labels and title
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric)
                ax.set_title(f'Model Comparison - {metric}')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()

                # Save the figure
                output_path = self.output_dir / f"comparison_{metric}.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

                plot_paths[metric] = output_path
                logger.info(
                    f"Created comparison plot for {metric} at {output_path}")

            except Exception as e:
                logger.error(
                    f"Error creating comparison plot for {metric}: {e}")

        return plot_paths

    @staticmethod
    def create_metrics_table(results_paths: List[Union[str, Path]],
                             model_names: List[str],
                             metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """Create a table comparing final metrics across models.

        Args:
            results_paths: List of paths to results.csv files
            model_names: List of model names
            metrics: List of metrics to include (if None, uses common metrics)

        Returns:
            DataFrame with metrics comparison
        """
        if len(results_paths) != len(model_names):
            raise ValueError(
                "Number of results paths must match number of model names")

        # Load all results
        final_metrics = []

        for path, name in zip(results_paths, model_names):
            try:
                df = pd.read_csv(Path(path))
                last_row = df.iloc[-1].to_dict()
                last_row['model'] = name
                final_metrics.append(last_row)
            except Exception as e:
                logger.error(f"Error loading results from {path}: {e}")
                continue

        if not final_metrics:
            logger.warning("No metrics could be loaded for comparison")
            return pd.DataFrame()

        # Create comparison dataframe
        comparison_df = pd.DataFrame(final_metrics)

        # Filter metrics if specified
        if metrics:
            cols_to_keep = ['model'] + \
                [col for col in comparison_df.columns if col in metrics]
            comparison_df = comparison_df[cols_to_keep]

        return comparison_df
