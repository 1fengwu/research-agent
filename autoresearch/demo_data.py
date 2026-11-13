"""AutoResearch 角色 B：固定 seed 的演示量价面板生成器。

用法：python -m autoresearch.demo_data

产物：
- data/demo_panel.csv      完整演示面板（固定 seed 42）
- data/demo_panel.csv.sha256
- data/sample.csv          给 D 前端联调的更小样本（固定 seed 7）
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .data import (
    build_profile,
    canonical_panel_hash,
    compute_labels,
    file_sha256,
    make_split,
    write_split_json,
)


def generate_demo_panel(
    *,
    seed: int = 42,
    symbols: int = 14,
    trading_days: int = 320,
    start: str = "2022-01-03",
    state_persistence: float = 0.85,
    state_scale: float = 0.5,
    alpha_scale: float = 0.008,
    noise_scale: float = 0.01,
    market_scale: float = 0.005,
) -> pd.DataFrame:
    """生成包含一点短期动量信号与量价耦合的合成面板。"""
    rng = random.Random(seed)
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start=start, periods=trading_days)]
    market_returns = [rng.gauss(0.0, market_scale) for _ in range(trading_days)]
    rows: List[Dict] = []

    for i in range(symbols):
        code = "S%03d" % (i + 1)
        price = round(rng.uniform(8.0, 120.0), 4)
        base_volume = rng.uniform(1.0e5, 1.0e6)
        state = 0.0
        for day_index, day in enumerate(dates):
            state = state_persistence * state + rng.gauss(0.0, state_scale)
            daily_return = (
                market_returns[day_index]
                + alpha_scale * state
                + rng.gauss(0.0, noise_scale)
            )
            close = max(price * (1.0 + daily_return), 0.5)
            volume_factor = 1.0 + 0.25 * abs(state) + rng.gauss(0.0, 0.06)
            volume = int(max(1000.0, base_volume * volume_factor))
            rows.append(
                {
                    "date": day,
                    "symbol": code,
                    "close": round(close, 4),
                    "volume": volume,
                }
            )
            price = close

    return pd.DataFrame(rows, columns=["date", "symbol", "close", "volume"])


def write_demo_artifacts(data_dir=None) -> Dict[str, str]:
    data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    demo_path = data_dir / "demo_panel.csv"
    sample_path = data_dir / "sample.csv"
    hash_path = data_dir / "demo_panel.csv.sha256"
    split_path = data_dir / "demo_split.json"
    profile_path = data_dir / "demo_profile.json"

    demo = generate_demo_panel(seed=42, symbols=14, trading_days=320, start="2022-01-03")
    sample = generate_demo_panel(seed=7, symbols=5, trading_days=80, start="2023-01-02")
    demo.to_csv(demo_path, index=False)
    sample.to_csv(sample_path, index=False)

    labelled = compute_labels(demo, horizon=5)
    split = make_split(
        labelled,
        seed=42,
        data_hash=canonical_panel_hash(demo),
        split_id="demo-split",
    )
    write_split_json(split, split_path)
    profile = build_profile(demo, horizon=5)
    with open(profile_path, "w", encoding="utf-8") as fh:
        import json

        json.dump(profile, fh, ensure_ascii=False, sort_keys=True, indent=2)
        fh.write("\n")

    digest = file_sha256(demo_path)
    hash_path.write_text(digest + "\n", encoding="utf-8")
    (data_dir / "README.md").write_text(
        "本目录是 AutoResearch 角色 B 的固定 seed 演示数据。\n\n"
        "- demo_panel.csv：完整演示面板，14 个标的，320 个交易日；固定 seed=42。\n"
        "- demo_panel.csv.sha256：文件字节的 sha256；运行期内容指纹使用 canonical_panel_hash() 计算。\n"
        "- demo_split.json：固定 seed=42 的时间切分、指纹与分段日期。\n"
        "- demo_profile.json：固定演示面板的数据画像。\n"
        "- sample.csv：5 个标的、80 个交易日的小样本，供 D 前端联调。\n",
        encoding="utf-8",
    )
    return {
        "demo": str(demo_path),
        "sample": str(sample_path),
        "sha256": digest,
        "sha256_file": str(hash_path),
        "split": str(split_path),
        "profile": str(profile_path),
        "readme": str(data_dir / "README.md"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 AutoResearch 演示量价面板")
    parser.add_argument("--data-dir", default=None, help="输出目录，默认 framework/data")
    args = parser.parse_args()
    artifacts = write_demo_artifacts(args.data_dir)
    for key, value in artifacts.items():
        print("%s=%s" % (key, value))


if __name__ == "__main__":
    main()
