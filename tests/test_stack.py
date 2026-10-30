"""Import and configuration checks for the pinned research stack.

These are engineering checks only: they show that the pinned versions import,
that BenchMARL's MAPPO/GRU configuration objects can be built, and that a model
can run a single forward pass on random tensors. They say nothing about
learning, task performance, or the proposed experience-sharing method.

The tests pass unchanged on a machine without the stack: every stack-dependent
class is skipped when ``benchmarl`` is not importable, which keeps the
lightweight CI job (no torch) meaningful.
"""

import dataclasses
import importlib.metadata
import pathlib
import unittest

try:
    import benchmarl
    import tensordict
    import torch
    import torchrl
    from benchmarl.algorithms import MappoConfig
    from benchmarl.models import GruConfig, MlpConfig
    from tensordict import TensorDict
    from torchrl.data import Categorical, Composite, UnboundedContinuous

    HAVE_STACK = True
except ImportError:  # pragma: no cover - exercised only without the stack
    HAVE_STACK = False

requires_stack = unittest.skipUnless(
    HAVE_STACK, "research stack not installed; run scripts/create_env.sh"
)

# Pins validated on macOS arm64 with CPython 3.11 (docs/development.md).
EXPECTED_VERSIONS = {
    "torch": "2.10.0",
    "torchvision": "0.25.0",
    "tensordict": "0.11.0",
    "torchrl": "0.11.1",
    "benchmarl": "1.5.2",
    "av": "13.1.0",
    "hydra-core": "1.3.2",
}

N_AGENTS = 3
OBS_DIM = 8
N_ACTIONS = 2
BATCH, SEQ = 2, 4


@requires_stack
class StackVersionTest(unittest.TestCase):
    """The environment must match the pins; a silent upgrade invalidates them."""

    def test_declared_versions(self):
        for package, expected in EXPECTED_VERSIONS.items():
            with self.subTest(package=package):
                self.assertEqual(importlib.metadata.version(package), expected)

    def test_torchrl_accepts_installed_tensordict(self):
        # BenchMARL requires torchrl>=0.10,<0.12, which in turn pins
        # tensordict>=0.11,<0.12. A mismatched pair imports but breaks at run time.
        from packaging.version import Version

        td_version = Version(tensordict.__version__)
        self.assertGreaterEqual(td_version, Version("0.11.0"))
        self.assertLess(td_version, Version("0.12.0"))

    def test_av_imports(self):
        # av is an unconditional BenchMARL dependency, not only for video logging.
        import av

        self.assertTrue(av.__version__.startswith("13."))


@requires_stack
class MappoConfigTest(unittest.TestCase):
    """MAPPO is the reference policy; its default config must still construct."""

    def test_default_config_fields(self):
        config = MappoConfig.get_from_yaml()
        self.assertEqual(config.clip_epsilon, 0.2)
        self.assertEqual(config.lmbda, 0.9)
        self.assertTrue(config.share_param_critic)
        self.assertEqual(config.scale_mapping, "biased_softplus_1.0")

    def test_flags(self):
        self.assertTrue(MappoConfig.on_policy())
        self.assertTrue(MappoConfig.has_centralized_critic())
        self.assertEqual(MappoConfig.associated_class().__name__, "Mappo")

    def test_config_is_editable(self):
        config = dataclasses.replace(
            MappoConfig.get_from_yaml(), share_param_critic=False, lmbda=0.95
        )
        self.assertFalse(config.share_param_critic)
        self.assertEqual(config.lmbda, 0.95)


@requires_stack
class ModelConstructionTest(unittest.TestCase):
    """Build one recurrent policy and one MLP critic from config, then forward."""

    def _policy_specs(self):
        input_spec = Composite(
            {
                "agents": Composite(
                    {"observation": UnboundedContinuous(shape=(N_AGENTS, OBS_DIM))},
                    shape=(N_AGENTS,),
                )
            }
        )
        output_spec = Composite(
            {
                "agents": Composite(
                    {
                        "logits": UnboundedContinuous(
                            shape=(N_AGENTS, N_ACTIONS)
                        )
                    },
                    shape=(N_AGENTS,),
                )
            }
        )
        return input_spec, output_spec

    def test_gru_policy_forward(self):
        torch.manual_seed(0)
        input_spec, output_spec = self._policy_specs()
        config = dataclasses.replace(
            GruConfig.get_from_yaml(), hidden_size=16, mlp_num_cells=[16]
        )
        model = config.get_model(
            input_spec=input_spec,
            output_spec=output_spec,
            agent_group="agents",
            input_has_agent_dim=True,
            n_agents=N_AGENTS,
            centralised=False,
            share_params=True,
            device="cpu",
            action_spec=Composite({"agents": Categorical(n=N_ACTIONS)}),
        )
        self.assertTrue(config.is_rnn)

        batch = TensorDict(
            {
                "agents": TensorDict(
                    {"observation": torch.randn(BATCH, SEQ, N_AGENTS, OBS_DIM)},
                    [BATCH, SEQ, N_AGENTS],
                ),
                # BenchMARL marks episode starts with a sequence-level flag.
                "is_init": torch.ones(BATCH, SEQ, 1, dtype=torch.bool),
            },
            [BATCH, SEQ],
        )
        first = model(batch.clone())["agents", "logits"]
        second = model(batch.clone())["agents", "logits"]

        self.assertEqual(tuple(first.shape), (BATCH, SEQ, N_AGENTS, N_ACTIONS))
        # With h_0=None the GRU starts from zeros, so the same input must
        # produce the same output; a mismatch means hidden state is leaking
        # between calls.
        torch.testing.assert_close(first, second)

    def test_mlp_critic_forward(self):
        torch.manual_seed(0)
        state_dim = 16
        config = dataclasses.replace(MlpConfig.get_from_yaml(), num_cells=[16])
        model = config.get_model(
            input_spec=Composite({"state": UnboundedContinuous(shape=(state_dim,))}),
            output_spec=Composite({"state_value": UnboundedContinuous(shape=(1,))}),
            agent_group="agents",
            input_has_agent_dim=False,
            n_agents=N_AGENTS,
            centralised=True,
            share_params=True,
            device="cpu",
            action_spec=Composite({"state": UnboundedContinuous(shape=(state_dim,))}),
        )
        batch = TensorDict({"state": torch.randn(BATCH, state_dim)}, [BATCH])
        value = model(batch)["state_value"]
        self.assertEqual(tuple(value.shape), (BATCH, 1))
        self.assertFalse(torch.isnan(value).any())


@requires_stack
class HydraConfigTest(unittest.TestCase):
    """The shipped Hydra experiment config must remain loadable and editable."""

    def test_base_experiment_config_loads(self):
        from omegaconf import OmegaConf

        root = pathlib.Path(benchmarl.__file__).parent
        path = root / "conf" / "experiment" / "base_experiment.yaml"
        self.assertTrue(path.exists(), path)
        config = OmegaConf.load(path)
        # Values recorded here are BenchMARL defaults, not project decisions.
        self.assertEqual(config.share_policy_params, True)
        self.assertEqual(config.gamma, 0.99)
        self.assertIn("on_policy_n_minibatch_iters", config)


if __name__ == "__main__":
    unittest.main()
