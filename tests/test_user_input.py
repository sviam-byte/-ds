"""Тесты парсинга пользовательского конфига и пресетов вариантов."""

from src.core.variant_presets import expand_variants
from src.io.user_input import build_run_spec, parse_user_input


def test_parse_user_input_key_value() -> None:
    cfg = parse_user_input("preset=causal; max_lag=10; window_sizes=128,256")
    assert cfg["preset"] == "causal"
    assert cfg["max_lag"] == "10"
    assert cfg["window_sizes"] == "128,256"


def test_parse_user_input_json() -> None:
    cfg = parse_user_input('{"preset":"basic","max_lag":7}')
    assert cfg == {"preset": "basic", "max_lag": 7}


def test_build_run_spec_defaults_and_lists() -> None:
    spec = build_run_spec(
        {
            "variants": "mutinf_full, te_directed",
            "window_sizes": "256,512",
            "window_stride": "64",
        },
        default_max_lag=12,
    )
    assert spec.variants == ["mutinf_full", "te_directed"]
    assert spec.max_lag == 12
    assert spec.window_sizes == [256, 512]
    assert spec.window_stride == 64


def test_expand_variants_preset_and_unique() -> None:
    variants, explain = expand_variants(["basic", "te_directed", "basic"])
    assert "preset 'basic'" in explain
    assert variants.count("corr_full") == 1
    assert variants.count("te_directed") == 1
