"""第5回の取得結果を使って、質問文から検索方式を選ぶ。"""

from collections.abc import Iterable
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Literal
import unicodedata

from app.config import ROOT

Route = Literal["vector", "graph"]
TRAINING_DATA = ROOT / "lectures" / "05" / "router_training.json"


def question_features(question: str) -> tuple[bool, ...]:
    """複数の問い、比較、手順、理由、具体例や例外の有無を順に返す。"""
    question = unicodedata.normalize("NFKC", question)
    return (
        question.count("?") >= 2 or bool(re.search(r"それぞれ|また[,、]|併せて|あわせて", question)),
        bool(re.search(r"違い|どう違|比較|共通点|使い分け", question)),
        bool(re.search(r"手順|流れ|順番|ステップ|段階|前後", question)),
        bool(re.search(r"理由|なぜ|どうして|目的|メリット|欠点", question)),
        bool(re.search(r"実例|具体例|例えば|たとえば|例外", question)),
    )


@dataclass(frozen=True)
class NaiveBayesRouter:
    priors: dict[Route, float]
    likelihoods: dict[Route, tuple[float, ...]]

    def predict(self, question: str) -> Route:
        features = question_features(question)
        scores = {
            route: math.log(prior) + sum(
                math.log(probability if present else 1 - probability)
                for present, probability in zip(features, self.likelihoods[route], strict=True)
            )
            for route, prior in self.priors.items()
        }
        if math.isclose(scores["graph"], scores["vector"], rel_tol=0, abs_tol=1e-12):
            return "vector"
        return "graph" if scores["graph"] > scores["vector"] else "vector"


def fit_router(examples: Iterable[tuple[str, Route]]) -> NaiveBayesRouter:
    """ラプラス平滑化を使うベルヌーイ・ナイーブベイズを学習する。"""
    by_route: dict[Route, list[tuple[bool, ...]]] = {"vector": [], "graph": []}
    for question, route in examples:
        by_route[route].append(question_features(question))

    if not all(by_route.values()):
        raise ValueError("学習データに vector と graph のラベルが各1件以上必要です。")

    total = sum(len(rows) for rows in by_route.values())
    priors = {route: (len(rows) + 1) / (total + len(by_route)) for route, rows in by_route.items()}
    likelihoods = {
        route: tuple((sum(values) + 1) / (len(rows) + 2) for values in zip(*rows, strict=True))
        for route, rows in by_route.items()
    }
    return NaiveBayesRouter(priors, likelihoods)


def load_router(path: Path = TRAINING_DATA) -> NaiveBayesRouter:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("ルーターの学習データには質問の配列が必要です。")

    examples: list[tuple[str, Route]] = []
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or "question" not in record or "route" not in record:
            raise ValueError(f"学習データの{index}件目に question と route が必要です。")
        question, route = record["question"], record["route"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"学習データの{index}件目の question に空でない文字列が必要です。")
        if route == "vector" or route == "graph":
            examples.append((question, route))
        elif route is not None:
            raise ValueError(f"学習データの{index}件目の route は vector、graph、null を指定してください。")
    return fit_router(examples)
