# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.v1.trainer_base import PPOTrainer, _hf_model_artifact_digest

BEHAVIOR_POLICY_IDENTITY = {
    "policy_version": 7,
    "weight_digest": "publication-7",
    "sampling_config_digest": "sampling",
    "runtime_identity": "runtime",
}


def _checkpoint_dir(checkpoint_root: Path) -> Path:
    return checkpoint_root / "global_step_7" / "actor" / "huggingface"


def _write_hf_checkpoint(checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "config.json").write_text(
        json.dumps({"architectures": ["Rwkv7ForCausalLM"], "model_type": "rwkv7"}),
        encoding="utf-8",
    )
    (checkpoint_dir / "model.safetensors").write_bytes(b"weights")


def test_hf_model_artifact_digest_accepts_complete_directory_and_tracks_file_content(tmp_path):
    checkpoint_dir = tmp_path / "actor" / "huggingface"
    _write_hf_checkpoint(checkpoint_dir)

    initial_digest = _hf_model_artifact_digest(checkpoint_dir)
    (checkpoint_dir / "model.safetensors").write_bytes(b"updated-weights")
    updated_digest = _hf_model_artifact_digest(checkpoint_dir)

    assert len(initial_digest) == 64
    assert len(updated_digest) == 64
    assert updated_digest != initial_digest


def test_hf_model_artifact_digest_rejects_incomplete_directory(tmp_path):
    checkpoint_dir = tmp_path / "actor" / "huggingface"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Incomplete Hugging Face model output.*no complete model weights"):
        _hf_model_artifact_digest(checkpoint_dir)


def _runtime_metadata(
    address: str,
    port: int,
    *,
    max_num_seqs: int,
    max_num_batched_tokens: int,
) -> dict[str, object]:
    return {
        "http_address": address,
        "http_port": port,
        "http_endpoint": f"{address}:{port}",
        "behavior_policy_identity": dict(BEHAVIOR_POLICY_IDENTITY),
        "weight_update_state": "active",
        "wkv_mode": "fp32io16",
        "vllm_version": "0.23.1.dev0",
        "capacity": {
            "capacity_source": "vllm.scheduler_config",
            "capacity_mode": "recurrent-state-no-kv-cache",
            "kv_cache_applicable": False,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_model_len": 10240,
            "gpu_memory_utilization": 0.8,
        },
    }


def _result_payload(
    pool_payload: dict[str, object],
    *,
    metrics: dict[str, float] | None = None,
) -> dict[str, object]:
    digest = hashlib.sha256(
        json.dumps(
            pool_payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 2,
        "weight_sha256": pool_payload["checkpoint_sha256"],
        "wkv_mode": pool_payload["wkv_mode"],
        "pool_manifest_lineage": {
            "manifest": copy.deepcopy(pool_payload),
            "manifest_sha256": digest,
        },
        "metrics": metrics or {"aime24/pass@1": 0.5},
    }


def test_external_evaluation_uses_checkpoint_command_and_result(tmp_path, monkeypatch):
    weight_root = tmp_path / "weights"
    checkpoint_root = weight_root / "maxrl" / "run"
    checkpoint_dir = _checkpoint_dir(checkpoint_root)
    events = []

    def save_checkpoint():
        _write_hf_checkpoint(checkpoint_dir)
        events.append("save")

    def run(command, *, cwd, env, check):
        assert command == ["helicopter", "eval", "--config", "maxrl.toml"]
        assert cwd == "/product/helicopter"
        assert check is True
        assert env["MAXRL_EVAL_WEIGHT"] == "maxrl/run/global_step_7/actor/huggingface"
        assert env["MAXRL_EVAL_STEP"] == "7"
        result_path = checkpoint_dir.parent / "lighteval_metrics.json"
        assert env["MAXRL_EVAL_RESULT_PATH"] == str(result_path)
        pool_path = Path(env["HELICOPTER_VLLM_POOL_MANIFEST"])
        assert pool_path.is_file()
        assert pool_path.stat().st_mode & 0o777 == 0o600
        pool_payload = json.loads(pool_path.read_text(encoding="utf-8"))
        assert pool_payload == {
            "schema_version": 3,
            "global_step": 7,
            "checkpoint_sha256": _hf_model_artifact_digest(checkpoint_dir),
            "checkpoint_display_name": "huggingface",
            "policy_checkpoint_path": "maxrl/run/global_step_7/actor/huggingface",
            "behavior_policy_identity": BEHAVIOR_POLICY_IDENTITY,
            "wkv_mode": "fp32io16",
            "vllm_version": "0.23.1.dev0",
            "max_model_len": 10240,
            "aggregate_scheduler_capacity": {
                "max_num_seqs": 160,
                "max_num_batched_tokens": 12288,
            },
            "replicas": [
                {
                    "base_url": "http://10.0.0.1:18000",
                    "behavior_policy_identity": BEHAVIOR_POLICY_IDENTITY,
                    "weight_update_state": "active",
                    "wkv_mode": "fp32io16",
                    "capacity_source": "vllm.scheduler_config",
                    "capacity_mode": "recurrent-state-no-kv-cache",
                    "kv_cache_applicable": False,
                    "max_num_seqs": 64,
                    "max_num_batched_tokens": 4096,
                    "max_model_len": 10240,
                    "gpu_memory_utilization": 0.8,
                },
                {
                    "base_url": "http://10.0.0.2:18001",
                    "behavior_policy_identity": BEHAVIOR_POLICY_IDENTITY,
                    "weight_update_state": "active",
                    "wkv_mode": "fp32io16",
                    "capacity_source": "vllm.scheduler_config",
                    "capacity_mode": "recurrent-state-no-kv-cache",
                    "kv_cache_applicable": False,
                    "max_num_seqs": 96,
                    "max_num_batched_tokens": 8192,
                    "max_model_len": 10240,
                    "gpu_memory_utilization": 0.8,
                },
            ],
        }
        result_path.write_text(
            json.dumps(
                _result_payload(
                    pool_payload,
                    metrics={
                        "aime24/pass@1": 0.5,
                        "math_500/pass@1": 0.25,
                    },
                )
            ),
            encoding="utf-8",
        )
        events.append("command")

    monkeypatch.setenv("WEIGHT_PATH", str(weight_root))
    monkeypatch.setenv("VLLM_RWKV7_WKV_MODE", "fp32io16")
    monkeypatch.setenv("HELICOPTER_PRODUCT_ROOT", "/product/helicopter")
    monkeypatch.setattr("verl.trainer.ppo.v1.trainer_base.subprocess.run", run)
    runtime_metadata = [
        _runtime_metadata(
            "10.0.0.1",
            18000,
            max_num_seqs=64,
            max_num_batched_tokens=4096,
        ),
        _runtime_metadata(
            "10.0.0.2",
            18001,
            max_num_seqs=96,
            max_num_batched_tokens=8192,
        ),
    ]
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "trainer": {"default_local_dir": str(checkpoint_root)},
                "data": {
                    "max_prompt_length": 2048,
                    "max_response_length": 8192,
                },
                "actor_rollout_ref": {
                    "rollout": {
                        "max_model_len": 10240,
                        "max_num_seqs": 64,
                    }
                },
            }
        ),
        global_steps=7,
        policy_round=SimpleNamespace(
            published=SimpleNamespace(as_dict=lambda: dict(BEHAVIOR_POLICY_IDENTITY))
        ),
        llm_server_manager=SimpleNamespace(
            get_addresses=lambda: ["10.0.0.1:18000", "10.0.0.2:18001"],
            get_runtime_metadata_snapshot=lambda: runtime_metadata,
        ),
        _save_checkpoint=save_checkpoint,
    )

    metrics = PPOTrainer._validate_external(
        trainer,
        OmegaConf.create({"command": ["helicopter", "eval", "--config", "maxrl.toml"]}),
    )

    assert metrics == {
        "val-core/aime24/pass@1": 0.5,
        "val-core/math_500/pass@1": 0.25,
    }
    assert events == ["save", "command"]
    assert not list(checkpoint_dir.parent.glob(".vllm-eval-pool-*.json"))


def test_external_evaluation_rejects_incomplete_hf_directory_after_checkpoint_save(tmp_path):
    checkpoint_root = tmp_path / "weights" / "maxrl" / "run"
    checkpoint_dir = _checkpoint_dir(checkpoint_root)
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
    save_calls = []
    trainer = SimpleNamespace(
        config=OmegaConf.create({"trainer": {"default_local_dir": str(checkpoint_root)}}),
        global_steps=7,
        _save_checkpoint=lambda: save_calls.append("save"),
    )

    with pytest.raises(
        RuntimeError,
        match=r"external evaluation checkpoint is incomplete: .*actor/huggingface",
    ) as error:
        PPOTrainer._validate_external(
            trainer,
            OmegaConf.create({"command": ["helicopter", "eval"]}),
        )

    assert save_calls == ["save"]
    assert isinstance(error.value.__cause__, RuntimeError)
    assert "no complete model weights" in str(error.value.__cause__)


def test_external_evaluation_rejects_stale_replica_policy_before_command(tmp_path, monkeypatch):
    weight_root = tmp_path / "weights"
    checkpoint_root = weight_root / "maxrl" / "run"
    checkpoint_dir = _checkpoint_dir(checkpoint_root)
    _write_hf_checkpoint(checkpoint_dir)
    stale = _runtime_metadata(
        "10.0.0.1",
        18000,
        max_num_seqs=64,
        max_num_batched_tokens=4096,
    )
    stale["behavior_policy_identity"] = {
        **BEHAVIOR_POLICY_IDENTITY,
        "policy_version": 6,
    }
    monkeypatch.setenv("WEIGHT_PATH", str(weight_root))
    monkeypatch.setenv("VLLM_RWKV7_WKV_MODE", "fp32io16")
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "trainer": {"default_local_dir": str(checkpoint_root)},
                "data": {"max_prompt_length": 2048, "max_response_length": 8192},
                "actor_rollout_ref": {"rollout": {}},
            }
        ),
        global_steps=7,
        policy_round=SimpleNamespace(
            published=SimpleNamespace(as_dict=lambda: dict(BEHAVIOR_POLICY_IDENTITY))
        ),
        llm_server_manager=SimpleNamespace(
            get_addresses=lambda: ["10.0.0.1:18000"],
            get_runtime_metadata_snapshot=lambda: [stale],
        ),
        _save_checkpoint=lambda: None,
    )

    with pytest.raises(RuntimeError, match="does not match the published policy"):
        PPOTrainer._validate_external(
            trainer,
            OmegaConf.create({"command": ["helicopter", "eval"]}),
        )


def test_external_evaluation_rejects_missing_published_policy_with_contract_error(tmp_path, monkeypatch):
    weight_root = tmp_path / "weights"
    checkpoint_root = weight_root / "maxrl" / "run"
    checkpoint_dir = _checkpoint_dir(checkpoint_root)
    _write_hf_checkpoint(checkpoint_dir)
    monkeypatch.setenv("WEIGHT_PATH", str(weight_root))
    monkeypatch.setenv("VLLM_RWKV7_WKV_MODE", "fp32io16")
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "trainer": {"default_local_dir": str(checkpoint_root)},
                "data": {"max_prompt_length": 2048, "max_response_length": 8192},
                "actor_rollout_ref": {"rollout": {}},
            }
        ),
        global_steps=7,
        llm_server_manager=SimpleNamespace(get_addresses=lambda: ["10.0.0.1:18000"]),
        _save_checkpoint=lambda: None,
    )

    with pytest.raises(RuntimeError, match="authoritative published behavior-policy identity"):
        PPOTrainer._validate_external(
            trainer,
            OmegaConf.create({"command": ["helicopter", "eval"]}),
        )


@pytest.mark.parametrize(
    "mismatch",
    [
        "missing_lineage",
        "checkpoint_sha256",
        "global_step",
        "behavior_policy_identity",
        "endpoint",
        "wkv_mode",
        "capacity",
        "manifest_digest",
    ],
)
def test_external_evaluation_rejects_result_lineage_mismatch_before_metrics(
    tmp_path,
    monkeypatch,
    mismatch,
):
    weight_root = tmp_path / "weights"
    checkpoint_root = weight_root / "maxrl" / "run"
    checkpoint_dir = _checkpoint_dir(checkpoint_root)
    _write_hf_checkpoint(checkpoint_dir)
    runtime_metadata = _runtime_metadata(
        "10.0.0.1",
        18000,
        max_num_seqs=64,
        max_num_batched_tokens=4096,
    )

    def run(_command, *, cwd, env, check):
        assert cwd == "/product/helicopter"
        assert check is True
        pool_payload = json.loads(
            Path(env["HELICOPTER_VLLM_POOL_MANIFEST"]).read_text(
                encoding="utf-8"
            )
        )
        result = _result_payload(pool_payload)
        if mismatch == "missing_lineage":
            del result["pool_manifest_lineage"]
        elif mismatch == "manifest_digest":
            result["pool_manifest_lineage"]["manifest_sha256"] = "b" * 64
        else:
            returned = result["pool_manifest_lineage"]["manifest"]
            if mismatch == "checkpoint_sha256":
                returned["checkpoint_sha256"] = "b" * 64
            elif mismatch == "global_step":
                returned["global_step"] = 8
            elif mismatch == "behavior_policy_identity":
                returned["behavior_policy_identity"]["policy_version"] = 8
                returned["replicas"][0]["behavior_policy_identity"][
                    "policy_version"
                ] = 8
            elif mismatch == "endpoint":
                returned["replicas"][0]["base_url"] = "http://10.0.0.2:18000"
            elif mismatch == "wkv_mode":
                returned["wkv_mode"] = "fp16"
                returned["replicas"][0]["wkv_mode"] = "fp16"
            elif mismatch == "capacity":
                returned["replicas"][0]["max_num_seqs"] = 65
                returned["aggregate_scheduler_capacity"]["max_num_seqs"] = 65
            else:  # pragma: no cover - parameter list owns the decision table
                raise AssertionError(mismatch)
        Path(env["MAXRL_EVAL_RESULT_PATH"]).write_text(
            json.dumps(result),
            encoding="utf-8",
        )

    monkeypatch.setenv("WEIGHT_PATH", str(weight_root))
    monkeypatch.setenv("VLLM_RWKV7_WKV_MODE", "fp32io16")
    monkeypatch.setenv("HELICOPTER_PRODUCT_ROOT", "/product/helicopter")
    monkeypatch.setattr("verl.trainer.ppo.v1.trainer_base.subprocess.run", run)
    trainer = SimpleNamespace(
        config=OmegaConf.create(
            {
                "trainer": {"default_local_dir": str(checkpoint_root)},
                "data": {"max_prompt_length": 2048, "max_response_length": 8192},
                "actor_rollout_ref": {"rollout": {}},
            }
        ),
        global_steps=7,
        policy_round=SimpleNamespace(
            published=SimpleNamespace(
                as_dict=lambda: dict(BEHAVIOR_POLICY_IDENTITY)
            )
        ),
        llm_server_manager=SimpleNamespace(
            get_addresses=lambda: ["10.0.0.1:18000"],
            get_runtime_metadata_snapshot=lambda: [runtime_metadata],
        ),
        _save_checkpoint=lambda: None,
    )

    with pytest.raises(RuntimeError, match="lineage|weight SHA|WKV mode"):
        PPOTrainer._validate_external(
            trainer,
            OmegaConf.create({"command": ["helicopter", "eval"]}),
        )
