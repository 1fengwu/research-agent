"""Helix 赫力 角色 B：CSV 接入、数据画像、标签与时间切分。

职责边界：
- 只读原始 CSV 并输出干净的实验面板；
- 标签、切分、滚动特征全部使用 t 日截止的历史信息；
- 不读 holdout，不调用 LLM，不写任何 verdict。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

REQUIRED_COLUMNS = ("date", "symbol", "close", "volume")
COLUMN_ALIASES = {
    "date": ("date", "trade_date", "trading_date", "datetime", "time"),
    "symbol": ("symbol", "code", "ticker", "sec_code", "instrument"),
    "close": ("close", "adj_close", "adjclose", "close_price", "close_adj"),
    "volume": ("volume", "vol", "amount", "turnover_volume", "trading_volume"),
}

DEFAULT_SPLIT = {
    "train": 0.6,
    "val_windows": 2,
    "test": None,
    "gap_days": 5,
}

FEATURE_GROUPS = {
    "base": ("r5", "r20"),
    "+volatility": ("r5", "r20", "hist_volatility", "vol_ratio"),
    "+volume": ("r5", "r20", "rel_volume", "volume_ratio", "volume_change"),
    "+volatility_volume": (
        "r5",
        "r20",
        "hist_volatility",
        "vol_ratio",
        "rel_volume",
        "volume_ratio",
        "volume_change",
    ),
    "+momentum": ("r5", "r20", "r10", "r60", "ma_dist5", "ma_dist20"),
    "+cross": ("r5", "r20", "x_rank_r5", "x_rank_r20", "x_rank_volume"),
}


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """返回数据文件本身的 sha256，供 TaskSpec.data_hash 使用。"""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _resolve_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    lowered = {str(col).strip().lower(): str(col) for col in df.columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def normalize_panel(
    df: pd.DataFrame,
    column_map: Optional[Dict[str, Sequence[str]]] = None,
    source_path: Optional[str] = None,
) -> pd.DataFrame:
    """列名映射 + 必填校验 + 唯一性 + 排序，输出标准化面板。"""
    aliases = dict(COLUMN_ALIASES)
    if column_map:
        for key, values in column_map.items():
            if key in aliases and values:
                aliases[key] = tuple(values)

    mapping: Dict[str, str] = {}
    missing: list[str] = []
    for standard in REQUIRED_COLUMNS:
        col = _resolve_column(df, aliases[standard])
        if col is None:
            missing.append(standard)
        else:
            mapping[col] = standard

    if missing:
        raise ValueError(
            "CSV 缺少必填列: %s（支持列名映射 %s）"
            % (", ".join(missing), json.dumps(aliases, ensure_ascii=False))
        )

    out = df.rename(columns=mapping).copy()
    keep = list(out.columns)
    for col in ("date", "symbol", "close", "volume"):
        if col not in keep:
            raise ValueError("标准化后缺少列 %s" % col)

    dates = pd.to_datetime(out["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("date 列存在无法解析的日期")
    out["date"] = dates

    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    if out["close"].isna().all():
        raise ValueError("close 列没有可用数值")
    if out["volume"].isna().all():
        raise ValueError("volume 列没有可用数值")
    if out["symbol"].isna().any():
        raise ValueError("symbol 列存在空值")

    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    dup = out.duplicated(subset=["date", "symbol"])
    if dup.any():
        first = out.loc[dup].head(3)[["date", "symbol"]].to_dict("records")
        raise ValueError("(date, symbol) 必须唯一，发现重复: %s" % first)

    out.attrs["source_path"] = source_path
    return out


def read_panel(
    path: Path,
    column_map: Optional[Dict[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    return normalize_panel(df, column_map=column_map, source_path=str(path))


def canonical_panel_hash(df: pd.DataFrame) -> str:
    """按标准列与排序做内容哈希，行序/额外列变化不影响判定。"""
    body = df[["date", "symbol", "close", "volume"]].copy()
    body["date"] = body["date"].astype(str)
    body = body.sort_values(["date", "symbol"]).reset_index(drop=True)
    return _sha256_text(body.to_csv(index=False))


def compute_labels(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """生成 fwd5 收益与同日截面中位数标签。

    严格大于中位数记 1，其余记 0；缺 t+5 的样本 label 为 NaN 并计数。
    """
    if horizon <= 0:
        raise ValueError("horizon 必须 > 0")

    out = df.sort_values(["symbol", "date"]).reset_index(drop=True).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("date 列存在无法解析的日期")
    grouped = out.groupby("symbol", sort=False)
    future_close = grouped["close"].transform(lambda s: s.shift(-horizon))
    out["fwd5_ret"] = future_close / out["close"] - 1.0
    out["label_horizon"] = horizon

    median = out.groupby("date", sort=True)["fwd5_ret"].transform("median")
    label = (out["fwd5_ret"] > median).astype(float)
    label[out["fwd5_ret"].isna()] = float("nan")
    out["label"] = label
    out["label_excluded"] = out["fwd5_ret"].isna().astype(int)
    return out


def build_profile(df: pd.DataFrame, horizon: int = 5) -> Dict[str, Any]:
    """数据画像：日期范围、标的数、缺失率、标签可计算比例与类别平衡。"""
    labelled = compute_labels(df, horizon=horizon)
    missing_rates = {
        col: round(float(labelled[col].isna().mean()), 6) if len(labelled) else 0.0
        for col in ("close", "volume")
    }
    computable = labelled["label"].notna()
    positive = 0
    if int(computable.sum()):
        positive = int(labelled.loc[computable, "label"].sum())
    total = int(computable.sum())

    return {
        "rows": int(len(labelled)),
        "date_start": str(labelled["date"].min().date()),
        "date_end": str(labelled["date"].max().date()),
        "symbols": int(labelled["symbol"].nunique()),
        "symbol_list": sorted(labelled["symbol"].unique().astype(str).tolist()),
        "label_horizon": int(horizon),
        "label_computable": total,
        "label_excluded": int(len(labelled) - total),
        "label_computable_ratio": round(total / len(labelled), 6) if len(labelled) else 0.0,
        "class_balance": {
            "negative": int(total - positive),
            "positive": positive,
            "positive_ratio": round(positive / total, 6) if total else 0.0,
        },
        "missing_rates": missing_rates,
        "data_hash": canonical_panel_hash(labelled),
    }


def describe_data(df: pd.DataFrame, horizon: int = 5) -> Dict[str, Any]:
    """A 的角色接口别名：返回数据画像。"""
    return build_profile(df, horizon=horizon)


@dataclass
class SplitResult:
    """切分结果：带 split 列的 DataFrame + 可复现的切分元数据。"""

    frame: pd.DataFrame
    meta: Dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return str(self.meta["fingerprint"])


def make_split(
    labelled_df: pd.DataFrame,
    train_ratio: float = DEFAULT_SPLIT["train"],
    val_windows: int = DEFAULT_SPLIT["val_windows"],
    test_ratio: Optional[float] = DEFAULT_SPLIT["test"],
    gap_days: int = DEFAULT_SPLIT["gap_days"],
    seed: int = 42,
    data_hash: Optional[str] = None,
    split_id: str = "split-001",
) -> SplitResult:
    """按日期整组切分，每个分段尾部按 gap_days 个交易日清除跨界样本。"""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio 必须在 (0, 1)")
    if val_windows < 2:
        raise ValueError("val_windows 必须 >= 2")
    if gap_days < 0:
        raise ValueError("gap_days 不能为负")

    if test_ratio is None:
        test_ratio = round((1.0 - train_ratio) / (val_windows + 1), 6)

    df = labelled_df.copy().sort_values(["date", "symbol"]).reset_index(drop=True)
    if "date" not in df or "label" not in df:
        raise ValueError("切分需要先 compute_labels()")

    dates = sorted(df["date"].unique().tolist())
    n_dates = len(dates)
    test_n = int(round(n_dates * test_ratio))
    remaining = n_dates - test_n
    train_n = int(round(remaining * train_ratio))
    val_remaining = n_dates - train_n - test_n
    if val_remaining < val_windows:
        raise ValueError("日期数量太少，无法形成 %d 个验证窗" % val_windows)

    val_base = val_remaining // val_windows
    counts = [train_n]
    for i in range(val_windows):
        if i < val_windows - 1:
            counts.append(val_base)
        else:
            counts.append(val_remaining - val_base * (val_windows - 1))
    counts.append(test_n)
    if sum(counts) != n_dates:
        raise ValueError("切分比例计算失败: %s -> %d" % (counts, sum(counts)))

    segment_names = ["train"] + ["val%d" % (i + 1) for i in range(val_windows)] + ["test"]
    date_to_segment: Dict[pd.Timestamp, str] = {}
    cursor = 0
    for name, count in zip(segment_names, counts):
        for day in dates[cursor : cursor + count]:
            date_to_segment[day] = name
        cursor += count

    included_dates: set[pd.Timestamp] = set()
    excluded_tail_dates: list[pd.Timestamp] = []
    for name in segment_names:
        seg_dates = [day for day in dates if date_to_segment[day] == name]
        keep_n = max(0, len(seg_dates) - gap_days)
        included_dates.update(seg_dates[:keep_n])
        excluded_tail_dates.extend(seg_dates[keep_n:])

    out = df.copy()
    out["split"] = out["date"].map(
        lambda day: date_to_segment.get(day, "excluded") if day in included_dates else "excluded"
    )

    segment_dates = {
        name: [d.strftime("%Y-%m-%d") for d in dates if date_to_segment[d] == name]
        for name in segment_names
    }
    segment_row_counts = {name: int((out["split"] == name).sum()) for name in segment_names}
    resolved_data_hash = data_hash or canonical_panel_hash(df)
    fingerprint = _sha256_text(
        json.dumps(
            {
                "split_id": split_id,
                "data_hash": resolved_data_hash,
                "seed": seed,
                "train_ratio": train_ratio,
                "test_ratio": test_ratio,
                "gap_days": gap_days,
                "val_windows": val_windows,
                "segment_dates": {name: segment_dates[name] for name in segment_names},
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )

    meta = {
        "split_id": split_id,
        "schema_version": "autoresearch-split-v1",
        "seed": seed,
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
        "val_windows": val_windows,
        "gap_days": gap_days,
        "date_counts": {name: len(segment_dates[name]) for name in segment_names},
        "row_counts": segment_row_counts,
        "excluded_tail_rows": int((out["split"] == "excluded").sum()),
        "excluded_tail_dates": [d.strftime("%Y-%m-%d") for d in excluded_tail_dates],
        "segment_dates": {name: segment_dates[name] for name in segment_names},
        "data_hash": resolved_data_hash,
        "fingerprint": fingerprint,
    }
    return SplitResult(frame=out, meta=meta)


def split_document(split: SplitResult) -> Dict[str, Any]:
    """生成可落盘的 split.json：元数据 + 指纹 + 每个分段实际日期。"""
    doc = dict(split.meta)
    doc["segments"] = {}
    for name in split.meta["row_counts"]:
        dates = (
            split.frame.loc[split.frame["split"] == name, "date"]
            .dt.strftime("%Y-%m-%d")
            .unique()
            .tolist()
        )
        doc["segments"][name] = dates
    return doc


def write_split_json(split: SplitResult, path) -> None:
    import json as _json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        _json.dump(split_document(split), fh, ensure_ascii=False, sort_keys=True, indent=2)
        fh.write("\n")


def split_segment(split: SplitResult, name: str = "train") -> pd.DataFrame:
    if name == "excluded":
        return split.frame[split.frame["split"] == "excluded"].copy()
    if name not in split.meta["row_counts"]:
        raise ValueError("未知分段 %s" % name)
    return split.frame[split.frame["split"] == name].copy()


def add_features(df: pd.DataFrame, feature_set: str = "base") -> pd.DataFrame:
    """按标的计算量价特征，任何滚动窗口都只使用历史与当日。"""
    if feature_set not in FEATURE_GROUPS:
        raise ValueError(
            "未知特征组 %s，可选: %s" % (feature_set, ", ".join(sorted(FEATURE_GROUPS)))
        )

    out = df.sort_values(["symbol", "date"]).reset_index(drop=True).copy()
    grouped = out.groupby("symbol", sort=False)

    out["r1"] = grouped["close"].transform(lambda s: s.pct_change(1))
    out["r5"] = grouped["close"].transform(lambda s: s.pct_change(5))
    out["r10"] = grouped["close"].transform(lambda s: s.pct_change(10))
    out["r20"] = grouped["close"].transform(lambda s: s.pct_change(20))
    out["r60"] = grouped["close"].transform(lambda s: s.pct_change(60))
    out["ma_dist5"] = (
        out["close"] / grouped["close"].transform(lambda s: s.rolling(5).mean()) - 1.0
    )
    out["ma_dist20"] = (
        out["close"] / grouped["close"].transform(lambda s: s.rolling(20).mean()) - 1.0
    )
    out["hist_volatility"] = grouped["r1"].transform(lambda s: s.rolling(20).std())
    out["vol_ratio"] = (
        out["hist_volatility"] / grouped["r1"].transform(lambda s: s.rolling(60).std())
    )
    out["x_rank_r5"] = out.groupby("date")["r5"].rank(pct=True)
    out["x_rank_r20"] = out.groupby("date")["r20"].rank(pct=True)
    out["x_rank_volume"] = out.groupby("date")["volume"].rank(pct=True)
    out["volume_ma20"] = grouped["volume"].transform(
        lambda s: s.rolling(20).mean().shift(1)
    )
    out["volume_ma5"] = grouped["volume"].transform(
        lambda s: s.rolling(5).mean().shift(1)
    )
    out["rel_volume"] = out["volume"] / out["volume_ma20"]
    out["volume_ratio"] = out["volume"] / out["volume_ma5"]
    out["volume_change"] = grouped["volume"].transform(lambda s: s.pct_change(1))

    columns = list(FEATURE_GROUPS[feature_set])
    out.attrs["feature_meta"] = {
        "feature_set": feature_set,
        "columns": columns,
        "fingerprint": feature_fingerprint(feature_set),
    }
    return out


def feature_fingerprint(feature_set: str) -> str:
    if feature_set not in FEATURE_GROUPS:
        raise ValueError("未知特征组 %s" % feature_set)
    body = json.dumps(
        {
            "schema": "autoresearch-features-v1",
            "feature_set": feature_set,
            "columns": list(FEATURE_GROUPS[feature_set]),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(body)
