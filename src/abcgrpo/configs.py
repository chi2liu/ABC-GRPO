# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

import trl


# TODO: add the shared options with a mixin to reduce code duplication
@dataclass
class GHPOConfig(trl.GRPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )
    hub_model_revision: Optional[str] = field(
        default="main", metadata={"help": "The Hub model branch to push the model to."}
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )

    # GSPO-reverse specific parameters
    clipped_token_penalty: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply a penalty to clipped tokens instead of ignoring them. When enabled, "
            "tokens that would be clipped get a reverse gradient update to push the model away from "
            "extreme probability ratios."
        },
    )
    clipped_token_penalty_gspo: bool = field(
        default=True,
        metadata={
            "help": "When clipped_token_penalty is enabled, use GSPO-style sequence-level reverse update. "
            "Clipped tokens switch to sequence-level importance sampling with reverse gradient direction. "
            "This provides stronger correction for systematic policy deviations."
        },
    )
    gspo_epsilon_low: Optional[float] = field(
        default=None,
        metadata={
            "help": "Lower bound epsilon for GSPO mode clipping. If None, uses the same value as epsilon. "
            "This controls how much the sequence-level coefficient can decrease (e.g., 0.3 means clip at 0.7)."
        },
    )
    gspo_epsilon_high: Optional[float] = field(
        default=None,
        metadata={
            "help": "Upper bound epsilon for GSPO mode clipping. If None, uses the same value as epsilon_high. "
            "This controls how much the sequence-level coefficient can increase (e.g., 0.5 means clip at 1.5). "
            "Can be set larger than epsilon to allow more flexibility in sequence-level corrections."
        },
    )

    # GSPO-to-GRPO fallback parameters
    gspo_to_grpo_fallback: bool = field(
        default=False,
        metadata={
            "help": "Enable GSPO-to-GRPO fallback mode. When enabled, sequences clipped at GSPO level "
            "will fallback to token-level GRPO updates instead of being discarded."
        },
    )
    gspo_fallback_epsilon: Optional[float] = field(
        default=None,
        metadata={
            "help": "Epsilon for token-level clipping when sequences fallback from GSPO to GRPO. "
            "If None, uses the main epsilon value. Can be set tighter than main epsilon."
        },
    )
    gspo_fallback_epsilon_high: Optional[float] = field(
        default=None,
        metadata={
            "help": "Upper bound epsilon for token-level clipping in GRPO fallback mode. "
            "If None, uses gspo_fallback_epsilon or epsilon_high."
        },
    )

    # GSPO Filter + GRPO Hybrid parameters
    gspo_filter_grpo_hybrid: bool = field(
        default=False,
        metadata={
            "help": "Enable GSPO Filter + GRPO Hybrid mode. First apply GSPO sequence-level clipping to filter out "
            "and discard clipped sequences, then apply GRPO token-level optimization to non-clipped sequences. "
            "For tokens clipped in GRPO, switch them to GSPO sequence-level coefficients (PPO-style)."
        },
    )
    gspo_filter_epsilon_low: Optional[float] = field(
        default=None,
        metadata={
            "help": "Lower-bound epsilon for GSPO filtering step in hybrid mode. If not specified, defaults to "
            "epsilon. This determines which sequences are discarded at GSPO level (e.g., 0.25 means discard if < 0.75)."
        },
    )
    gspo_filter_epsilon_high: Optional[float] = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon for GSPO filtering step in hybrid mode. If not specified, defaults to "
            "epsilon_high or epsilon. This determines which sequences are discarded (e.g., 0.25 means discard if > 1.25)."
        },
    )
    grpo_token_epsilon_low: Optional[float] = field(
        default=None,
        metadata={
            "help": "Lower-bound epsilon for GRPO token-level clipping in hybrid mode. If not specified, defaults to "
            "epsilon. Used for token-level clipping on non-discarded sequences (e.g., 0.1 means clip at 0.9)."
        },
    )
    grpo_token_epsilon_high: Optional[float] = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon for GRPO token-level clipping in hybrid mode. If not specified, defaults to "
            "epsilon_high or epsilon. Used for token-level clipping (e.g., 0.1 means clip at 1.1)."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        # Set default GSPO epsilon values if not specified
        if self.gspo_epsilon_low is None:
            self.gspo_epsilon_low = self.epsilon
        if self.gspo_epsilon_high is None:
            self.gspo_epsilon_high = self.epsilon_high if self.epsilon_high is not None else self.epsilon

        # Set default GSPO fallback epsilon values if not specified
        if self.gspo_fallback_epsilon is None:
            self.gspo_fallback_epsilon = self.epsilon
        if self.gspo_fallback_epsilon_high is None:
            self.gspo_fallback_epsilon_high = self.gspo_fallback_epsilon

        # Set default GSPO Filter + GRPO Hybrid epsilon values if not specified
        if self.gspo_filter_grpo_hybrid:
            if self.gspo_filter_epsilon_low is None:
                self.gspo_filter_epsilon_low = self.epsilon
            if self.gspo_filter_epsilon_high is None:
                self.gspo_filter_epsilon_high = self.epsilon_high if self.epsilon_high is not None else self.epsilon
            if self.grpo_token_epsilon_low is None:
                self.grpo_token_epsilon_low = self.epsilon
            if self.grpo_token_epsilon_high is None:
                self.grpo_token_epsilon_high = self.epsilon_high if self.epsilon_high is not None else self.epsilon


@dataclass
class GHPOHybridConfig(GHPOConfig):
    """
    Configuration class specifically for GSPO Filter + GRPO Hybrid mode.
    Inherits from GHPOConfig and sets appropriate defaults for hybrid training.
    """

    def __post_init__(self):
        # Enable hybrid mode by default
        self.gspo_filter_grpo_hybrid = True

        # Set recommended defaults if not specified
        if self.gspo_filter_epsilon_low is None:
            self.gspo_filter_epsilon_low = 0.25  # Looser for filtering
        if self.gspo_filter_epsilon_high is None:
            self.gspo_filter_epsilon_high = 0.25
        if self.grpo_token_epsilon_low is None:
            self.grpo_token_epsilon_low = 0.1  # Tighter for token control
        if self.grpo_token_epsilon_high is None:
            self.grpo_token_epsilon_high = 0.1

        # Call parent post_init
        super().__post_init__()

        # Log configuration
        import logging
        logger = logging.getLogger(__name__)
        logger.info("GSPO Filter + GRPO Hybrid mode configured:")
        logger.info(f"  GSPO filter epsilon: [{self.gspo_filter_epsilon_low}, {self.gspo_filter_epsilon_high}]")
        logger.info(f"  GRPO token epsilon: [{self.grpo_token_epsilon_low}, {self.grpo_token_epsilon_high}]")


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": ("The entity to store runs under.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": ("The project to store runs under.")},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": ("The group to store runs under.")},
    )


@dataclass
class GHPOScriptArguments(trl.ScriptArguments):
    """
    Script arguments for the GHPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', 'tag_count', 'code', 'ioi_code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash", "cpp"],
        },
    )
    code_eval_test_batch_size: int = field(
        default=1,
        metadata={
            "help": "for each generation, evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases. Useful to avoid overloading the eval server + save time on wrong solutions"
        },
    )
    parallel_code_exec_per_proc: int = field(
        default=2,
        metadata={
            "help": "Number of parallel E2B code executions per process. Default of 2 is suitable for the Free Hobby tier of E2B with 8 GPUs used for training."
        },
    )

    dataset_prompt_column: str = field(
        default="prompt",
        metadata={"help": "Column to use as prompts for training."},
    )

    e2b_router_url: Optional[str] = field(
        default=None,
        metadata={"help": "URL for the E2B route. See scripts/e2b_router.py"},
    )
    
    rl_type: Optional[str] = field(
        default="grpo",
        metadata={"help": "Type of RL algorithm to use: 'grpo', 'abcgrpo', 'grpo_adaptive', etc."},
    )
