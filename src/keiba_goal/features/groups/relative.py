"""レース内相対 (段階 B): 元の特徴量の field_dec 上の順位と z 値。

元の特徴量は config/default.yaml の features.relative.base ("グループ.特徴量") に並べる (import 時に既定の設定から読む)。
順位は field_dec の馬のうち値のある馬の中の昇順 (同値は平均順位)、z = (値 − 平均) / 標準偏差 (母標準偏差)。
field_dec 外の馬・値の無い馬は NaN。値のある馬が 2 頭未満のレースの z は NaN、全頭同じ値なら z = 0。
"""

from __future__ import annotations

import polars as pl

from keiba_goal.config import load_config
from keiba_goal.features.context import Ctx
from keiba_goal.features.exprs import f32
from keiba_goal.features.registry import Group, RegistryError, spec_maker

NAME = "relative"
S = spec_maker(NAME, "B")


def _base() -> tuple[tuple[str, str], ...]:
    base = (load_config().section("features").get("relative") or {}).get("base")
    if not base:
        raise RegistryError("config features.relative.base (\"グループ.特徴量\" の並び) がありません")
    pairs = tuple(tuple(b.split(".", 1)) for b in base)
    if bad := [b for b, p in zip(base, pairs) if len(p) != 2]:
        raise RegistryError(f"features.relative.base は \"グループ.特徴量\": {bad}")
    return pairs  # type: ignore[return-value]


BASE = _base()
INPUTS = tuple(f for _, f in BASE)
SPECS = tuple(
    spec
    for f in INPUTS
    for spec in (
        S(f"rel_rank_{f}", "num", ["in_field_dec"], doc=f"{f} の field_dec 内の順位 (昇順)"),
        S(f"rel_z_{f}", "num", ["in_field_dec"], doc=f"{f} の field_dec 内の z 値"),
    )
)


def build(ctx: Ctx) -> pl.DataFrame:
    rows = ctx.target("in_field_dec").join(ctx.features(*INPUTS), on="race_id", how="left")
    exprs = []
    for f in INPUTS:
        v = pl.when(pl.col("in_field_dec") & pl.col(f).is_not_nan()).then(pl.col(f))
        n = v.count().over("race_key")
        std = v.std(ddof=0).over("race_key")
        exprs += [
            f32(v.rank("average").over("race_key"), f"rel_rank_{f}"),
            f32(
                pl.when(n < 2).then(None).when(std > 0).then((v - v.mean().over("race_key")) / std).otherwise(v * 0.0),
                f"rel_z_{f}",
            ),
        ]
    return rows.select("race_id", *exprs).collect()


GROUP = Group(NAME, "B", SPECS, build, requires=tuple(sorted({g for g, _ in BASE})), inputs=INPUTS)
