"""
Training utilities for MAGI.
"""

from .train import (
    build_default_callbacks,
    compile_model,
    fit_model,
    train_single_run,
)

from .resume import (
    ResumeCheckpoint,
    find_checkpoint,
    load_resume_state,
)

from .prior_refit import (
    PriorRefitResult,
    encode_posterior,
    posterior_kl,
    prior_posterior_auc,
    refit_prior,
    refit_prior_run,
    save_refit_model,
)

from .adaptive_callbacks import (
    TaskAdaptiveLossScheduler,
    TaskAdaptiveTrainingMonitor,
    ValidationEnergyDistributionMonitor,
)

from .checkpointing import (
    build_model_metadata,
    save_training_checkpoint,
    save_final_trained_model,
    extract_callback_metadata,
    load_metadata,
    load_json,
    load_task_adaptive_model_for_generation,
)

__all__ = [
    "ResumeCheckpoint",
    "find_checkpoint",
    "load_resume_state",
    "PriorRefitResult",
    "encode_posterior",
    "posterior_kl",
    "prior_posterior_auc",
    "refit_prior",
    "refit_prior_run",
    "save_refit_model",
    "build_default_callbacks",
    "compile_model",
    "fit_model",
    "train_single_run",
    "TaskAdaptiveLossScheduler",
    "TaskAdaptiveTrainingMonitor",
    "ValidationEnergyDistributionMonitor",
    "build_model_metadata",
    "load_json",
    "load_metadata",
    "load_task_adaptive_model_for_generation",
    "save_training_checkpoint",
    "save_final_trained_model",
    "extract_callback_metadata",
]