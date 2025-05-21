import numpy as np
from transformers import AutoConfig

import sglang as sgl

@sgl.function
def image_qa(s, image_path, question):
    s += sgl.image(image_path) + question
    s += sgl.gen("action")

class TokenActionConverter:
    def __init__(self, n_action_bins: int = 256, unnorm_key: str = "bridge_orig"):
        self.bins = np.linspace(-1, 1, n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.vocab_size = 32000
        self.unnorm_key = unnorm_key
        self.config = AutoConfig.from_pretrained(
            "openvla/openvla-7b", trust_remote_code=True
        ).to_dict()
        self.norm_stats = self.config["norm_stats"]
        assert unnorm_key is not None
        if unnorm_key not in self.norm_stats:
            raise ValueError(
                f"The `unnorm_key` you chose ({unnorm_key = }) is not in the available statistics. "
                f"Please choose from: {self.norm_stats.keys()}"
            )

    def token_to_action(self, output_ids):
        """
        Convert token IDs to actions.

        Args:
            output_ids (list or np.ndarray): Token IDs to convert

        Returns:
            np.ndarray: The corresponding actions
        """
        predicted_action_token_ids = np.array(output_ids)
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.norm_stats[self.unnorm_key]["action"]
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(
            action_norm_stats["q01"]
        )
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) *
            (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions

    def action_to_token(self, actions):
        """
        Convert actions back to token IDs.

        Args:
            actions (np.ndarray): The actions to convert

        Returns:
            np.ndarray: The corresponding token IDs
        """
        # First, normalize the actions back to [-1, 1] range
        action_norm_stats = self.norm_stats[self.unnorm_key]["action"]
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(
            action_norm_stats["q01"]
        )

        # Reverse the unnormalization
        normalized_actions = np.where(
            mask,
            2 * (actions - action_low) / (action_high - action_low) - 1,
            actions
        )

        # Find the closest bin centers to the normalized actions
        discretized_actions = np.array([
            np.abs(self.bin_centers - val).argmin()
            for val in normalized_actions
        ])

        # Convert back to token ids
        output_ids = self.vocab_size - discretized_actions - 1

        return output_ids