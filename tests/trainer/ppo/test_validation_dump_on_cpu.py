import json

from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def test_write_generations_preserves_validation_diagnostics_on_cpu(tmp_path):
    RayPPOTrainer._write_generations(
        inputs=["prompt"],
        outputs=["answer"],
        gts=["ground truth"],
        scores=[1.0],
        reward_extra_infos_dict={
            "uid": ["sample-0"],
            "stop_reason": ["completed"],
            "response_token_count": [7],
            "repetition_truncated": [True],
            "repetition_matched_reason": ["zstd_low_ratio"],
            "sampling_params": [{"top_k": 40, "top_p": 0.35, "temperature": 0.25}],
        },
        dump_path=tmp_path,
        global_steps=0,
    )

    row = json.loads((tmp_path / "0.jsonl").read_text(encoding="utf-8").splitlines()[0])

    assert row["uid"] == "sample-0"
    assert row["stop_reason"] == "completed"
    assert row["response_token_count"] == 7
    assert row["repetition_truncated"] is True
    assert row["repetition_matched_reason"] == "zstd_low_ratio"
    assert row["sampling_params"] == {"top_k": 40, "top_p": 0.35, "temperature": 0.25}


def test_validation_sampling_params_for_dump_keeps_eval_overrides_on_cpu():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "response_length": 7168,
                    "val_kwargs": {
                        "n": 4,
                        "do_sample": True,
                        "temperature": 0.25,
                        "top_p": 0.35,
                        "top_k": 40,
                        "presence_penalty": 0.65,
                        "repetition_penalty": 0.25,
                        "penalty_decay": 0.99,
                        "logprobs": None,
                    },
                },
            },
        }
    )

    params = RayPPOTrainer._validation_sampling_params_for_dump(config)

    assert params["n"] == 4
    assert params["top_k"] == 40
    assert params["top_p"] == 0.35
    assert params["temperature"] == 0.25
    assert params["presence_penalty"] == 0.65
    assert params["repetition_penalty"] == 0.25
    assert params["penalty_decay"] == 0.99
    assert params["logprobs"] is None
    assert params["max_tokens"] == 7168
