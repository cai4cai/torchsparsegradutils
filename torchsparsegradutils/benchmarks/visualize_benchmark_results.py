#!/usr/bin/env python3
"""
Comprehensive visualization script for torc        # Set labels and title
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=45)
        ax.set_title(title)
        ax.set_ylabel(ylabel)

        # Always set logarithmic scale
        ax.set_yscale('log')radutils benchmark results.
Creates multiple types of plots grouped by different dimensions for paper and documentation use.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Set style for publication-quality plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


class BenchmarkVisualizer:
    def __init__(self, results_dir="/workspaces/torchsparsegradutils/torchsparsegradutils/benchmarks/results"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(
            "/workspaces/torchsparsegradutils/torchsparsegradutils/benchmarks/benchmark_visualizations"
        )
        self.output_dir.mkdir(exist_ok=True)

        # Load all CSV files
        self.data = {}
        self.load_all_data()

    def _add_failure_markers(self, ax, x_positions, time_values, algorithm_names=None):
        """Add red X markers for failed experiments (NaN, 0, or missing values)"""
        for i, (x_pos, time_val) in enumerate(zip(x_positions, time_values)):
            if pd.isna(time_val) or time_val == 0:
                # Get y position in log scale - use a small positive value
                y_pos = ax.get_ylim()[0] * 10  # Position at bottom of plot
                ax.scatter(x_pos, y_pos, marker="X", color="red", s=100, zorder=10)

    def _create_bar_plot_with_failures(self, ax, algorithms, times, stds=None, title="", ylabel="Time (μs)"):
        """Create bar plot with error bars and failure markers"""
        valid_mask = ~(pd.isna(times) | (times == 0))
        valid_indices = np.where(valid_mask)[0]
        failed_indices = np.where(~valid_mask)[0]

        # Plot valid bars
        if len(valid_indices) > 0:
            valid_times = times[valid_indices]
            valid_stds = stds[valid_indices] if stds is not None else None

            ax.bar(
                valid_indices, valid_times, yerr=valid_stds if valid_stds is not None else None, capsize=5, alpha=0.8
            )

        # Add failure markers
        if len(failed_indices) > 0:
            # Set y-axis to log scale first to establish limits
            if np.any(valid_mask):
                ax.set_yscale("log")
                # Use minimum valid value / 10 as failure marker position
                min_valid = np.min(times[valid_mask]) if np.any(valid_mask) else 1
                failure_y = min_valid / 10
            else:
                failure_y = 1  # fallback value

            for idx in failed_indices:
                ax.scatter(idx, failure_y, marker="X", color="red", s=150, zorder=10)

        # Set labels and title
        ax.set_xticks(range(len(algorithms)))
        ax.set_xticklabels(algorithms, rotation=45)
        ax.set_title(title)
        ax.set_ylabel(ylabel)

        # Always set logarithmic scale
        ax.set_yscale("log")

    def _create_memory_and_time_plots(self, combo_df, axes, show_residual=False):
        """Create standardized memory usage and computation time plots"""
        self._create_memory_plot(combo_df, axes[0])
        self._create_time_plot(combo_df, axes[1])
        if show_residual:
            self._create_residual_plot(combo_df, axes[2])

    def _create_memory_plot(self, combo_df, ax):
        """Create memory usage plot"""
        if "fwd_mem_MB" in combo_df.columns and "bwd_mem_MB" in combo_df.columns:
            mem_stats = (
                combo_df.groupby("algorithm")
                .agg({"fwd_mem_MB": ["mean", "std"], "bwd_mem_MB": ["mean", "std"]})
                .reset_index()
            )

            if not mem_stats.empty:
                algorithms = mem_stats["algorithm"].values
                fwd_mem = mem_stats[("fwd_mem_MB", "mean")].fillna(0).values
                bwd_mem = mem_stats[("bwd_mem_MB", "mean")].fillna(0).values
                fwd_mem_std = mem_stats[("fwd_mem_MB", "std")].fillna(0).values
                bwd_mem_std = mem_stats[("bwd_mem_MB", "std")].fillna(0).values

                self._plot_phase_bars(
                    ax,
                    algorithms,
                    fwd_mem,
                    bwd_mem,
                    fwd_mem_std,
                    bwd_mem_std,
                    "Algorithm",
                    "Memory (MB)",
                    "Memory Usage by Algorithm and Phase",
                )
            else:
                ax.text(0.5, 0.5, "No memory data available", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "No memory columns found", ha="center", va="center", transform=ax.transAxes)

    def _create_time_plot(self, combo_df, ax):
        """Create computation time plot"""
        if "fwd_time_us" in combo_df.columns and "bwd_time_us" in combo_df.columns:
            time_stats = (
                combo_df.groupby("algorithm")
                .agg({"fwd_time_us": ["mean", "std"], "bwd_time_us": ["mean", "std"]})
                .reset_index()
            )

            algorithms = time_stats["algorithm"].values
            fwd_time = time_stats[("fwd_time_us", "mean")].fillna(0).values
            bwd_time = time_stats[("bwd_time_us", "mean")].fillna(0).values
            fwd_time_std = time_stats[("fwd_time_us", "std")].fillna(0).values
            bwd_time_std = time_stats[("bwd_time_us", "std")].fillna(0).values

            self._plot_phase_bars(
                ax,
                algorithms,
                fwd_time,
                bwd_time,
                fwd_time_std,
                bwd_time_std,
                "Algorithm",
                "Time (μs)",
                "Computation Time by Algorithm and Phase",
            )

    def _create_residual_plot(self, combo_df, ax):
        """Create residual quality plot"""
        residual_col = "relative_resnorm" if "relative_resnorm" in combo_df.columns else "relative_residual_norm"
        if residual_col in combo_df.columns:
            quality_stats = combo_df.groupby("algorithm").agg({residual_col: ["mean", "std"]}).reset_index()
            quality_stats.columns = ["algorithm", "quality_mean", "quality_std"]

            quality_means = quality_stats["quality_mean"].fillna(0).values
            quality_stds = quality_stats["quality_std"].fillna(0).values
            quality_algos = quality_stats["algorithm"].values

            self._create_bar_plot_with_failures(
                ax,
                quality_algos,
                quality_means,
                quality_stds,
                "Solution Quality (Relative Residual)",
                "Relative Residual",
            )
            ax.set_yscale("log")
        else:
            ax.text(0.5, 0.5, "No residual column found", ha="center", va="center", transform=ax.transAxes)

    def _plot_phase_bars(self, ax, algorithms, fwd_data, bwd_data, fwd_std, bwd_std, xlabel, ylabel, title):
        """Plot forward and backward phase bars with failure markers"""
        x = np.arange(len(algorithms))
        width = 0.35

        # Forward phase bars
        valid_fwd = ~(pd.isna(fwd_data) | (fwd_data == 0))
        if np.any(valid_fwd):
            ax.bar(
                x[valid_fwd] - width / 2,
                fwd_data[valid_fwd],
                width,
                yerr=fwd_std[valid_fwd],
                label="Forward",
                alpha=0.8,
                capsize=3,
            )

        # Backward phase bars
        valid_bwd = ~(pd.isna(bwd_data) | (bwd_data == 0))
        if np.any(valid_bwd):
            ax.bar(
                x[valid_bwd] + width / 2,
                bwd_data[valid_bwd],
                width,
                yerr=bwd_std[valid_bwd],
                label="Backward",
                alpha=0.8,
                capsize=3,
            )

        # Add failure markers
        self._add_phase_failure_markers(ax, x, width, valid_fwd, valid_bwd, fwd_data, bwd_data)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=45)
        ax.set_yscale("log")
        ax.legend()

    def _add_phase_failure_markers(self, ax, x, width, valid_fwd, valid_bwd, fwd_data, bwd_data):
        """Add red X markers for failed phases"""
        failed_fwd = ~valid_fwd
        failed_bwd = ~valid_bwd

        if np.any(failed_fwd | failed_bwd):
            min_val = 1
            if np.any(valid_fwd):
                min_val = min(min_val, np.min(fwd_data[valid_fwd]))
            if np.any(valid_bwd):
                min_val = min(min_val, np.min(bwd_data[valid_bwd]))

            failure_y = min_val / 10
            for i in range(len(x)):
                if failed_fwd[i]:
                    ax.scatter(x[i] - width / 2, failure_y, marker="X", color="red", s=100, zorder=10)
                if failed_bwd[i]:
                    ax.scatter(x[i] + width / 2, failure_y, marker="X", color="red", s=100, zorder=10)

    def load_all_data(self):
        """Load all benchmark CSV files"""
        csv_files = list(self.results_dir.glob("*.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                benchmark_name = csv_file.stem.replace("_results", "")
                self.data[benchmark_name] = df
                print(f"Loaded {benchmark_name}: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")

    def create_performance_comparison_plots(self):
        """Create performance comparison plots for different algorithms"""

        # 1. Sparse Matrix Multiplication Performance (using suite data)
        if "sparse_mm_suite" in self.data:
            self._plot_sparse_mm_performance()

        # 2. Sparse Solve Performance (using suite data)
        if "sparse_generic_solve_suite" in self.data:
            self._plot_sparse_solve_performance()

        # 3. Triangular Solve Performance (using suitesparse data)
        if "sparse_triangular_solve_suitesparse" in self.data:
            self._plot_triangular_solve_performance()

        # 4. Suite Performance (Real matrices) - keeping this for any additional suite data
        self._plot_suite_performance()

    def _plot_sparse_mm_performance(self):
        """Plot sparse matrix multiplication performance split by dtype and layout combinations (using suite data)"""
        df = self.data["sparse_mm_suite"].copy()

        # Get all dtype and layout combinations
        combinations = []
        for index_dt in df["index_dt"].dropna().unique():
            for value_dt in df["value_dt"].dropna().unique():
                for layout in df["layout"].dropna().unique():
                    combinations.append((index_dt, value_dt, layout))

        # Create separate plots for each combination
        for idx_dt, val_dt, layout in combinations:
            # Filter data for this specific combination - include ALL data, even failed experiments
            combo_df = df[(df["index_dt"] == idx_dt) & (df["value_dt"] == val_dt) & (df["layout"] == layout)].copy()

            if len(combo_df) == 0:
                continue

            # Create 1x2 subplot layout: Memory Usage and Computation Time
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            combo_title = f"Sparse MM Performance: {idx_dt}/{val_dt}/{layout.upper()}"
            fig.suptitle(combo_title, fontsize=16, fontweight="bold")

            # Use the new standardized plotting function (no residual for MM)
            self._create_memory_and_time_plots(combo_df, axes, show_residual=False)

            plt.tight_layout()
            filename = f"sparse_mm_suite_performance_{idx_dt}_{val_dt}_{layout}.pdf"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"✓ Created {filename}")

    def _plot_sparse_solve_performance(self):
        """Plot sparse solve performance split by dtype and layout combinations (using suite data)"""
        df = self.data["sparse_generic_solve_suite"].copy()

        # Get all dtype and layout combinations that have data
        combinations = []
        for index_dt in df["index_dt"].dropna().unique():
            for value_dt in df["value_dt"].dropna().unique():
                for layout in df["layout"].dropna().unique():
                    combinations.append((index_dt, value_dt, layout))

        # Create separate plots for each combination
        for idx_dt, val_dt, layout in combinations:
            # Filter data for this specific combination - include ALL data, even failed experiments
            combo_df = df[(df["index_dt"] == idx_dt) & (df["value_dt"] == val_dt) & (df["layout"] == layout)].copy()

            if len(combo_df) == 0:
                continue

            # Create 1x3 subplot layout: Memory Usage, Computation Time, and Relative Residual
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            combo_title = f"Sparse Solve Performance: {idx_dt}/{val_dt}/{layout.upper()}"
            fig.suptitle(combo_title, fontsize=16, fontweight="bold")

            # Use the new standardized plotting function (with residual for sparse solve)
            self._create_memory_and_time_plots(combo_df, axes, show_residual=True)

            plt.tight_layout()
            filename = f"sparse_solve_suite_performance_{idx_dt}_{val_dt}_{layout}.pdf"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"✓ Created {filename}")

    def _plot_triangular_solve_performance(self):
        """Plot triangular solve performance split by dtype and layout combinations (using suitesparse data)"""
        df = self.data["sparse_triangular_solve_suitesparse"].copy()

        # Get all dtype and layout combinations
        combinations = []
        for index_dt in df["index_dt"].dropna().unique():
            for value_dt in df["value_dt"].dropna().unique():
                for layout in df["layout"].dropna().unique():
                    combinations.append((index_dt, value_dt, layout))

        # Create separate plots for each combination
        for idx_dt, val_dt, layout in combinations:
            # Filter data for this specific combination - include ALL data, even failed experiments
            combo_df = df[(df["index_dt"] == idx_dt) & (df["value_dt"] == val_dt) & (df["layout"] == layout)].copy()

            if len(combo_df) == 0:
                continue

            # Create 1x3 subplot layout: Memory Usage, Computation Time, and Relative Residual
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            combo_title = f"Triangular Solve Performance: {idx_dt}/{val_dt}/{layout.upper()}"
            fig.suptitle(combo_title, fontsize=16, fontweight="bold")

            # Use the new standardized plotting function (with residual for triangular solve)
            self._create_memory_and_time_plots(combo_df, axes, show_residual=True)

            plt.tight_layout()
            filename = f"triangular_solve_suitesparse_performance_{idx_dt}_{val_dt}_{layout}.pdf"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"✓ Created {filename}")

    def _plot_suite_performance(self):
        """Plot performance on real matrix suites"""
        suite_datasets = [name for name in self.data.keys() if "suite" in name]

        if not suite_datasets:
            return

        fig, axes = plt.subplots(len(suite_datasets), 2, figsize=(15, 6 * len(suite_datasets)))
        if len(suite_datasets) == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle("Real Matrix Suite Performance", fontsize=16, fontweight="bold")

        for i, dataset in enumerate(suite_datasets):
            df = self.data[dataset].copy()
            df = df.dropna(subset=["fwd_time_us"])

            if len(df) == 0:
                continue

            # Performance by algorithm
            if "algorithm" in df.columns:
                algo_col = "algorithm"
            else:
                algo_col = "algo"

            perf_data = df.groupby(algo_col)["fwd_time_us"].agg(["mean", "std"]).reset_index()
            perf_data = perf_data.sort_values("mean")

            axes[i, 0].barh(perf_data[algo_col], perf_data["mean"], xerr=perf_data["std"])
            axes[i, 0].set_title(f'{dataset.replace("_", " ").title()} - Forward Time')
            axes[i, 0].set_xlabel("Time (μs)")
            axes[i, 0].set_xscale("log")

            # Memory usage if available
            if "fwd_mem_MB" in df.columns:
                mem_data = df.groupby(algo_col)["fwd_mem_MB"].mean().sort_values()
                mem_data.plot(kind="barh", ax=axes[i, 1])
                axes[i, 1].set_title(f'{dataset.replace("_", " ").title()} - Memory Usage')
                axes[i, 1].set_xlabel("Memory (MB)")

        plt.tight_layout()
        plt.savefig(self.output_dir / "suite_performance.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_performance_scaling(self, scaling_datasets, axes, idx_dt, val_dt, layout):
        """Helper method to plot performance scaling by problem size"""
        for name, df in scaling_datasets:
            if all(col in df.columns for col in ["index_dt", "value_dt", "layout"]):
                combo_df = df[
                    (df["index_dt"] == idx_dt) & (df["value_dt"] == val_dt) & (df["layout"] == layout)
                ].dropna(subset=["fwd_time_us"])

                if len(combo_df) > 0 and "size" in df.columns:
                    size_scaling = combo_df.groupby("size")["fwd_time_us"].mean()
                    if len(size_scaling) > 1:
                        axes[0, 0].plot(size_scaling.index, size_scaling.values, "o-", label=name.replace("_", " "))

        axes[0, 0].set_title("Performance Scaling by Problem Size")
        axes[0, 0].set_ylabel("Forward Time (μs)")
        axes[0, 0].set_xlabel("Problem Size")
        axes[0, 0].set_yscale("log")
        axes[0, 0].legend()

    def _plot_nnz_scaling(self, scaling_datasets, axes, idx_dt, val_dt, layout):
        """Helper method to plot performance vs number of non-zeros"""
        for name, df in scaling_datasets:
            if all(col in df.columns for col in ["index_dt", "value_dt", "layout", "nnz"]):
                combo_df = df[
                    (df["index_dt"] == idx_dt) & (df["value_dt"] == val_dt) & (df["layout"] == layout)
                ].dropna(subset=["fwd_time_us", "nnz"])

                if len(combo_df) > 0:
                    sample_df = combo_df.sample(min(50, len(combo_df)))
                    axes[0, 1].scatter(
                        sample_df["nnz"], sample_df["fwd_time_us"], alpha=0.6, label=name.replace("_", " ")
                    )

        axes[0, 1].set_title("Performance vs Number of Non-zeros")
        axes[0, 1].set_xlabel("Number of Non-zeros")
        axes[0, 1].set_ylabel("Forward Time (μs)")
        axes[0, 1].set_xscale("log")
        axes[0, 1].set_yscale("log")
        axes[0, 1].legend()

    def _plot_memory_scaling(self, scaling_datasets, axes, idx_dt, val_dt, layout):
        """Helper method to plot memory usage scaling"""
        memory_data_found = False
        for name, df in scaling_datasets:
            if all(col in df.columns for col in ["index_dt", "value_dt", "layout", "fwd_mem_MB", "N"]):
                combo_df = df[
                    (df["index_dt"] == idx_dt) & (df["value_dt"] == val_dt) & (df["layout"] == layout)
                ].dropna(subset=["fwd_mem_MB", "N"])

                if len(combo_df) > 0:
                    mem_scaling = combo_df.groupby("N")["fwd_mem_MB"].mean()
                    if len(mem_scaling) > 1:
                        axes[1, 0].plot(
                            mem_scaling.index,
                            mem_scaling.values,
                            "o-",
                            label=f"{name.replace('_', ' ')} (n={len(combo_df)})",
                        )
                        memory_data_found = True

        axes[1, 0].set_title("Memory Usage Scaling (Valid Measurements Only)")
        axes[1, 0].set_xlabel("Matrix Size (N)")
        axes[1, 0].set_ylabel("Memory (MB)")
        axes[1, 0].set_xscale("log")
        axes[1, 0].set_yscale("log")

        if memory_data_found:
            axes[1, 0].legend()
        else:
            axes[1, 0].text(
                0.5,
                0.5,
                "No valid memory scaling data for this combination",
                ha="center",
                va="center",
                transform=axes[1, 0].transAxes,
            )

    def _plot_efficiency_analysis(self, scaling_datasets, axes, idx_dt, val_dt, layout):
        """Helper method to plot computational efficiency"""
        efficiency_data_found = False
        for name, df in scaling_datasets:
            if all(col in df.columns for col in ["index_dt", "value_dt", "layout", "nnz", "fwd_time_us"]):
                combo_df = df[
                    (df["index_dt"] == idx_dt) & (df["value_dt"] == val_dt) & (df["layout"] == layout)
                ].dropna(subset=["nnz", "fwd_time_us"])

                if len(combo_df) > 0:
                    combo_df_copy = combo_df.copy()
                    combo_df_copy["ops_per_us"] = combo_df_copy["nnz"] / combo_df_copy["fwd_time_us"]
                    sample_df = combo_df_copy.sample(min(30, len(combo_df_copy)))
                    axes[1, 1].scatter(
                        sample_df["nnz"], sample_df["ops_per_us"], alpha=0.6, label=name.replace("_", " ")
                    )
                    efficiency_data_found = True

        axes[1, 1].set_title("Computational Efficiency")
        axes[1, 1].set_xlabel("Number of Non-zeros")
        axes[1, 1].set_ylabel("Operations per μs")
        axes[1, 1].set_xscale("log")

        if efficiency_data_found:
            axes[1, 1].legend()
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No valid efficiency data for this combination",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )

    def create_scaling_analysis_plots(self):
        """Create plots showing how performance scales with problem size (using rand data for size variation)"""
        # Analyze datasets that have size information (primarily rand datasets)
        scaling_datasets = []
        for name, df in self.data.items():
            if "N" in df.columns and "fwd_time_us" in df.columns:
                scaling_datasets.append((name, df))

        if not scaling_datasets:
            return

        # Create scaling plots for each dtype/layout combination
        all_combinations = set()
        for name, df in scaling_datasets:
            if all(col in df.columns for col in ["index_dt", "value_dt", "layout"]):
                for idx_dt in df["index_dt"].dropna().unique():
                    for val_dt in df["value_dt"].dropna().unique():
                        for layout in df["layout"].dropna().unique():
                            all_combinations.add((idx_dt, val_dt, layout))

        for idx_dt, val_dt, layout in all_combinations:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            combo_title = f"Performance Scaling Analysis: {idx_dt}/{val_dt}/{layout.upper()}"
            fig.suptitle(combo_title, fontsize=16, fontweight="bold")

            # Use helper methods to create individual plots
            self._plot_performance_scaling(scaling_datasets, axes, idx_dt, val_dt, layout)
            self._plot_nnz_scaling(scaling_datasets, axes, idx_dt, val_dt, layout)
            self._plot_memory_scaling(scaling_datasets, axes, idx_dt, val_dt, layout)
            self._plot_efficiency_analysis(scaling_datasets, axes, idx_dt, val_dt, layout)

            plt.tight_layout()
            filename = f"scaling_analysis_rand_{idx_dt}_{val_dt}_{layout}.pdf"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"✓ Created {filename}")

    def _plot_dense_vs_sparse_comparison(self, suite_benchmarks, axes):
        """Helper method to plot dense vs sparse comparison"""
        dense_sparse_data = []
        for name, df in suite_benchmarks.items():
            if "algo" in df.columns or "algorithm" in df.columns:
                algo_col = "algo" if "algo" in df.columns else "algorithm"
                for _, row in df.iterrows():
                    algo_name = row[algo_col]
                    fwd_time = row.get("fwd_time_us", np.nan)

                    if "dense" in algo_name.lower():
                        dense_sparse_data.append({"benchmark": name, "approach": "Dense", "time": fwd_time})
                    elif "sparse" in algo_name.lower():
                        dense_sparse_data.append({"benchmark": name, "approach": "Sparse", "time": fwd_time})

        if dense_sparse_data:
            dense_sparse_df = pd.DataFrame(dense_sparse_data)
            valid_data = dense_sparse_df.dropna(subset=["time"])
            if len(valid_data) > 0:
                sns.boxplot(data=valid_data, x="benchmark", y="time", hue="approach", ax=axes[0, 0])
                axes[0, 0].set_title("Dense vs Sparse Approach Performance")
                axes[0, 0].set_ylabel("Forward Time (μs)")
                axes[0, 0].set_yscale("log")
                axes[0, 0].tick_params(axis="x", rotation=45)

    def _plot_layout_comparison(self, suite_benchmarks, axes):
        """Helper method to plot layout comparison (COO vs CSR)"""
        layout_data = []
        for name, df in suite_benchmarks.items():
            if "layout" in df.columns:
                for _, row in df.iterrows():
                    layout_data.append(
                        {"benchmark": name, "layout": row["layout"], "time": row.get("fwd_time_us", np.nan)}
                    )

        if layout_data:
            layout_df = pd.DataFrame(layout_data)
            valid_data = layout_df.dropna(subset=["time"])
            if len(valid_data) > 0:
                sns.boxplot(data=valid_data, x="benchmark", y="time", hue="layout", ax=axes[0, 1])
                axes[0, 1].set_title("Sparse Matrix Layout Performance")
                axes[0, 1].set_ylabel("Forward Time (μs)")
                axes[0, 1].set_yscale("log")
                axes[0, 1].tick_params(axis="x", rotation=45)

    def _plot_dtype_comparison(self, suite_benchmarks, axes):
        """Helper method to plot data type impact"""
        dtype_data = []
        for name, df in suite_benchmarks.items():
            if "value_dt" in df.columns:
                for _, row in df.iterrows():
                    dtype_data.append(
                        {"benchmark": name, "dtype": row["value_dt"], "time": row.get("fwd_time_us", np.nan)}
                    )

        if dtype_data:
            dtype_df = pd.DataFrame(dtype_data)
            valid_data = dtype_df.dropna(subset=["time"])
            if len(valid_data) > 0:
                sns.boxplot(data=valid_data, x="benchmark", y="time", hue="dtype", ax=axes[1, 0])
                axes[1, 0].set_title("Data Type Impact on Performance")
                axes[1, 0].set_ylabel("Forward Time (μs)")
                axes[1, 0].set_yscale("log")
                axes[1, 0].tick_params(axis="x", rotation=45)

    def _plot_backend_comparison(self, suite_benchmarks, axes):
        """Helper method to plot backend comparison (PyTorch vs CuPy vs JAX)"""
        backend_data = []
        for name, df in suite_benchmarks.items():
            algo_col = "algo" if "algo" in df.columns else "algorithm" if "algorithm" in df.columns else None
            if algo_col:
                for _, row in df.iterrows():
                    algo_name = row[algo_col]
                    backend = "Unknown"
                    if "cupy" in algo_name.lower():
                        backend = "CuPy"
                    elif "jax" in algo_name.lower():
                        backend = "JAX"
                    elif "torch" in algo_name.lower() or "dense" in algo_name.lower() or "sparse" in algo_name.lower():
                        backend = "PyTorch"

                    if backend != "Unknown":
                        backend_data.append(
                            {"benchmark": name, "backend": backend, "time": row.get("fwd_time_us", np.nan)}
                        )

        if backend_data:
            backend_df = pd.DataFrame(backend_data)
            valid_data = backend_df.dropna(subset=["time"])
            if len(valid_data) > 0:
                sns.boxplot(data=valid_data, x="benchmark", y="time", hue="backend", ax=axes[1, 1])
                axes[1, 1].set_title("Backend Performance Comparison")
                axes[1, 1].set_ylabel("Forward Time (μs)")
                axes[1, 1].set_yscale("log")
                axes[1, 1].tick_params(axis="x", rotation=45)

    def create_comparative_analysis_plots(self):
        """Create plots comparing different approaches across benchmarks (suite/suitesparse data only)"""
        # Only include suite or suitesparse benchmarks for fair comparison
        suite_benchmarks = {k: v for k, v in self.data.items() if "suite" in k.lower() or "suitesparse" in k.lower()}

        if not suite_benchmarks:
            print("⚠ No suite or suitesparse data available for comparative analysis")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Cross-Benchmark Comparative Analysis (Suite/SuiteSparse Data)", fontsize=16, fontweight="bold")

        # Use helper methods for each comparison plot
        self._plot_dense_vs_sparse_comparison(suite_benchmarks, axes)
        self._plot_layout_comparison(suite_benchmarks, axes)
        self._plot_dtype_comparison(suite_benchmarks, axes)
        self._plot_backend_comparison(suite_benchmarks, axes)

        plt.tight_layout()
        plt.savefig(self.output_dir / "comparative_analysis_suite.pdf", dpi=300, bbox_inches="tight")
        plt.close()
        print("✓ Created comparative_analysis_suite.pdf")

    def create_summary_report(self):
        """Create a summary report with key findings"""

        print("\n" + "=" * 80)
        print("BENCHMARK VISUALIZATION SUMMARY REPORT")
        print("=" * 80)

        for name, df in self.data.items():
            print(f"\n{name.upper().replace('_', ' ')}:")
            print(f"  - Total measurements: {len(df)}")

            if "fwd_time_us" in df.columns:
                fwd_times = df["fwd_time_us"].dropna()
                if len(fwd_times) > 0:
                    print(f"  - Forward time range: {fwd_times.min():.1f} - {fwd_times.max():.1f} μs")
                    print(f"  - Forward time median: {fwd_times.median():.1f} μs")

            if "algorithm" in df.columns or "algo" in df.columns:
                algo_col = "algorithm" if "algorithm" in df.columns else "algo"
                algorithms = df[algo_col].value_counts()
                print(f"  - Algorithms tested: {', '.join(algorithms.index[:5])}")
                if len(algorithms) > 5:
                    print(f"    (and {len(algorithms)-5} more)")

        print(f"\nVisualization files saved to: {self.output_dir}")
        print("Generated plots:")
        for pdf_file in self.output_dir.glob("*.pdf"):
            print(f"  - {pdf_file.name}")

        print("\nRecommended plots for different use cases:")
        print("  - Paper/Publication: comparative_analysis.pdf, scaling_analysis.pdf")
        print("  - Documentation: sparse_mm_performance.pdf, sparse_solve_performance.pdf")
        print("  - Detailed Analysis: All generated PDFs")

    def generate_all_visualizations(self):
        """Generate all visualization types"""
        print("Generating comprehensive benchmark visualizations...")

        self.create_performance_comparison_plots()
        print("✓ Performance comparison plots created")

        self.create_scaling_analysis_plots()
        print("✓ Scaling analysis plots created")

        self.create_comparative_analysis_plots()
        print("✓ Comparative analysis plots created")

        self.create_summary_report()


if __name__ == "__main__":
    visualizer = BenchmarkVisualizer()
    visualizer.generate_all_visualizations()
