#!/usr/bin/env python3

# Copyright 2025 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""
Demonstration of unsupervised object ID association learning in Monty.

This script shows how to set up and run experiments that test the ability of
learning modules to discover correspondences between their object representations
without requiring predefined object labels.
"""

import logging
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmarks.configs.unsupervised_association_experiments import CONFIGS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_simple_cross_modal_demo():
    """Run a simple cross-modal association learning demonstration."""
    logger.info("Starting simple cross-modal association learning demo...")

    # Get the experiment configuration
    config = CONFIGS['simple_cross_modal_association']

    # Create and run the experiment
    experiment = config['experiment_class'](config)

    logger.info("Running training phase...")
    experiment.train()

    logger.info("Running evaluation phase...")
    experiment.evaluate()

    # Get association statistics from the model
    if hasattr(experiment.model, 'get_association_summary'):
        summary = experiment.model.get_association_summary()
        logger.info(f"Association learning summary: {summary}")

        # Print detailed statistics for each learning module
        for lm_stats in summary['lm_statistics']:
            lm_id = lm_stats.get('lm_id', 'unknown')
            total_assoc = lm_stats.get('total_associations', 0)
            strong_assoc = lm_stats.get('strong_associations', 0)
            avg_strength = lm_stats.get('average_strength', 0.0)

            logger.info(f"LM {lm_id}: {total_assoc} total associations, "
                        f"{strong_assoc} strong, avg strength: {avg_strength:.3f}")

    logger.info("Simple cross-modal demo completed!")
    return experiment


def run_multi_modal_demo():
    """Run a multi-modal association learning demonstration."""
    logger.info("Starting multi-modal association learning demo...")

    # Get the experiment configuration
    config = CONFIGS['multi_modal_association']

    # Create and run the experiment
    experiment = config['experiment_class'](config)

    logger.info("Running training phase...")
    experiment.train()

    logger.info("Running evaluation phase...")
    experiment.evaluate()

    # Analyze cross-modal associations
    if hasattr(experiment.model, 'get_association_summary'):
        summary = experiment.model.get_association_summary()

        logger.info("Multi-modal association analysis:")
        logger.info(f"Total LMs: {summary['total_lms']}")

        cross_lm = summary['cross_lm_analysis']
        logger.info(f"Recent spatial consistency: {cross_lm['recent_spatial_consistency']:.3f}")
        logger.info(f"Recent confidence correlation: {cross_lm['recent_confidence_correlation']:.3f}")

        # Print association matrix
        logger.info("Association matrix between LMs:")
        for i, lm_stats in enumerate(summary['lm_statistics']):
            associations_by_lm = lm_stats.get('associations_by_lm', {})
            logger.info(f"LM {i} -> {dict(associations_by_lm)}")

    logger.info("Multi-modal demo completed!")
    return experiment


def log_detailed_associations(lm):
    """Log detailed associations for a learning module."""
    if not hasattr(lm, 'association_memory'):
        return
    logger.info("Detailed associations:")
    for my_obj_id in lm.association_memory:
        for other_lm_id in lm.association_memory[my_obj_id]:
            associated_objects = lm.get_associated_object_ids(my_obj_id, other_lm_id)
            if associated_objects:
                logger.info(f"  {my_obj_id} <-> {other_lm_id}: {associated_objects}")


def analyze_association_learning(experiment):
    """Analyze the association learning results in detail."""
    logger.info("Analyzing association learning results...")

    if not hasattr(experiment.model, 'learning_modules'):
        logger.warning("No learning modules found for analysis")
        return

    for i, lm in enumerate(experiment.model.learning_modules):
        if not hasattr(lm, 'get_association_statistics'):
            continue

        logger.info(f"\n=== Learning Module {i} Analysis ===")

        stats = lm.get_association_statistics()
        logger.info(f"Total associations: {stats['total_associations']}")
        logger.info(f"Strong associations: {stats['strong_associations']}")
        logger.info(f"Average strength: {stats['average_strength']:.3f}")

        # Analyze associations by other LM
        associations_by_lm = stats.get('associations_by_lm', {})
        for other_lm_id, count in associations_by_lm.items():
            logger.info(f"  -> {other_lm_id}: {count} associations")

        # Log detailed association information
        log_detailed_associations(lm)


def compare_association_strategies():
    """Compare different association learning strategies."""
    logger.info("Comparing association learning strategies...")

    strategies = ['simple_cross_modal_association', 'conservative_association']
    results = {}

    for strategy in strategies:
        logger.info(f"\nTesting strategy: {strategy}")

        config = CONFIGS[strategy]
        experiment = config['experiment_class'](config)

        # Run a shorter experiment for comparison
        experiment.train()

        # Collect results
        if hasattr(experiment.model, 'get_association_summary'):
            summary = experiment.model.get_association_summary()
            results[strategy] = summary

            # Print key metrics
            total_associations = sum(
                lm_stats.get('total_associations', 0)
                for lm_stats in summary['lm_statistics']
            )
            strong_associations = sum(
                lm_stats.get('strong_associations', 0)
                for lm_stats in summary['lm_statistics']
            )

            logger.info(f"  Total associations: {total_associations}")
            logger.info(f"  Strong associations: {strong_associations}")

            if total_associations > 0:
                strength_ratio = strong_associations / total_associations
                logger.info(f"  Strong association ratio: {strength_ratio:.3f}")

    # Compare results
    logger.info("\n=== Strategy Comparison ===")
    for strategy, summary in results.items():
        total_strong = sum(
            lm_stats.get('strong_associations', 0)
            for lm_stats in summary['lm_statistics']
        )
        logger.info(f"{strategy}: {total_strong} strong associations")


def main():
    """Main demonstration function."""
    logger.info("Unsupervised Object ID Association Learning Demo")
    logger.info("=" * 50)

    try:
        # Run simple cross-modal demo
        logger.info("\n1. Running simple cross-modal demonstration...")
        simple_experiment = run_simple_cross_modal_demo()

        # Analyze the results
        analyze_association_learning(simple_experiment)

        # Run multi-modal demo
        logger.info("\n2. Running multi-modal demonstration...")
        run_multi_modal_demo()

        # Compare strategies
        logger.info("\n3. Comparing association strategies...")
        compare_association_strategies()

        logger.info("\n" + "=" * 50)
        logger.info("Demo completed successfully!")

    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
