# Copyright 2025 Thousand Brains Project
# Copyright 2023-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import csv
import logging
import os
from collections import defaultdict

import numpy as np


class KNNProfiler:
    """Simple profiler for tracking KNN operations performance."""

    _instance = None

    @classmethod
    def get_instance(cls):
        """Singleton pattern to ensure we have a single profiler instance.

        Returns:
            Singleton instance of this class
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.stats = defaultdict(list)
        self.enabled = False

    def enable(self):
        """Enable profiling."""
        self.enabled = True

    def disable(self):
        """Disable profiling."""
        self.enabled = False

    def clear(self):
        """Clear all collected statistics."""
        self.stats = defaultdict(list)

    def record_build(self, backend, points_shape, elapsed_time):
        """Record statistics for index building operation.

        Args:
            backend: The backend used ('cpu' or 'gpu')
            points_shape: Shape of the points array
            elapsed_time: Time taken for the operation in seconds
        """
        if not self.enabled:
            return

        self.stats["build"].append(
            {
                "backend": backend,
                "n_points": points_shape[0],
                "dim": points_shape[1],
                "time": elapsed_time,
            }
        )

    def record_search(self, backend, query_shape, k, elapsed_time):
        """Record statistics for search operation.

        Args:
            backend: The backend used ('cpu' or 'gpu')
            query_shape: Shape of the query points array
            k: Number of neighbors requested
            elapsed_time: Time taken for the operation in seconds
        """
        if not self.enabled:
            return

        self.stats["search"].append(
            {
                "backend": backend,
                "n_queries": query_shape[0],
                "dim": query_shape[1],
                "k": k,
                "time": elapsed_time,
            }
        )

    def get_summary(self):
        """Get a summary of the collected statistics.

        Returns:
            A dictionary with summary statistics
        """
        summary = {}

        if self.stats.get("build"):
            build_times = [item["time"] for item in self.stats["build"]]
            build_points = [item["n_points"] for item in self.stats["build"]]
            cpu_build = [
                item for item in self.stats["build"] if item["backend"] == "cpu"
            ]
            gpu_build = [
                item for item in self.stats["build"] if item["backend"] == "gpu"
            ]

            summary["build"] = {
                "count": len(build_times),
                "total_time": sum(build_times),
                "avg_time": np.mean(build_times) if build_times else 0,
                "max_time": max(build_times) if build_times else 0,
                "min_time": min(build_times) if build_times else 0,
                "avg_points": np.mean(build_points) if build_points else 0,
                "cpu_count": len(cpu_build),
                "gpu_count": len(gpu_build),
                "cpu_avg_time": np.mean([item["time"] for item in cpu_build])
                if cpu_build
                else 0,
                "gpu_avg_time": np.mean([item["time"] for item in gpu_build])
                if gpu_build
                else 0,
            }

        if self.stats.get("search"):
            search_times = [item["time"] for item in self.stats["search"]]
            search_queries = [item["n_queries"] for item in self.stats["search"]]
            cpu_search = [
                item for item in self.stats["search"] if item["backend"] == "cpu"
            ]
            gpu_search = [
                item for item in self.stats["search"] if item["backend"] == "gpu"
            ]

            summary["search"] = {
                "count": len(search_times),
                "total_time": sum(search_times),
                "avg_time": np.mean(search_times) if search_times else 0,
                "max_time": max(search_times) if search_times else 0,
                "min_time": min(search_times) if search_times else 0,
                "avg_queries": np.mean(search_queries) if search_queries else 0,
                "cpu_count": len(cpu_search),
                "gpu_count": len(gpu_search),
                "cpu_avg_time": np.mean([item["time"] for item in cpu_search])
                if cpu_search
                else 0,
                "gpu_avg_time": np.mean([item["time"] for item in gpu_search])
                if gpu_search
                else 0,
            }

        return summary

    def save_to_csv(self, output_dir, run_name):
        """Save the collected statistics to CSV files.

        Args:
            output_dir: Directory to save the CSV files
            run_name: Name of the run to include in the filename
        """
        if not self.stats:
            logging.warning("No KNN profiling data to save")
            return

        os.makedirs(output_dir, exist_ok=True)

        if self.stats.get("build"):
            build_file = os.path.join(output_dir, f"{run_name}_knn_build_stats.csv")
            with open(build_file, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["backend", "n_points", "dim", "time"]
                )
                writer.writeheader()
                for item in self.stats["build"]:
                    writer.writerow(item)

        if self.stats.get("search"):
            search_file = os.path.join(output_dir, f"{run_name}_knn_search_stats.csv")
            with open(search_file, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["backend", "n_queries", "dim", "k", "time"]
                )
                writer.writeheader()
                for item in self.stats["search"]:
                    writer.writerow(item)

        # Save summary
        summary = self.get_summary()
        summary_file = os.path.join(output_dir, f"{run_name}_knn_summary.csv")

        with open(summary_file, "w", newline="") as f:
            f.write("KNN Operation Summary\n\n")

            if "build" in summary:
                f.write("Build Operations:\n")
                for key, value in summary["build"].items():
                    f.write(f"{key},{value}\n")
                f.write("\n")

            if "search" in summary:
                f.write("Search Operations:\n")
                for key, value in summary["search"].items():
                    f.write(f"{key},{value}\n")

        logging.info(f"KNN profiling data saved to {output_dir}")
