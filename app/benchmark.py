"""第1回 発展: 会話履歴なしでモデル×質問の応答時間・使用量を比較する。"""

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from statistics import mean, median
from time import perf_counter, sleep

import httpx
from google import genai
from google.genai import errors, types

from app.ask import DEFAULT_QUESTION
from app.config import MODEL, api_error_message, get_api_key

DEFAULT_QUESTIONS = [
    "LoRAとは何ですか？100字程度で説明してください。",
    DEFAULT_QUESTION + "100字程度で答えてください。",
    "RAGとファインチューニングの違いを500字程度で説明してください。",
]


@dataclass
class Result:
    started_at: str
    model: str
    question: str
    trial: int
    temperature: float
    max_output_tokens: int
    interval_seconds: float
    elapsed_seconds: float = 0.0
    status: str = "ok"
    http_status: int | None = None
    model_version: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    thoughts_tokens: int | None = None
    cached_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    answer: str = ""
    error: str = ""


def measure(
    client: genai.Client,
    model: str,
    question: str,
    trial: int,
    temperature: float,
    max_output_tokens: int,
    interval: float,
) -> Result:
    """API一回を計測する。失敗も記録し、使用量不明を0に置き換えない。"""
    result = Result(
        started_at=datetime.now(timezone.utc).isoformat(),
        model=model, question=question, trial=trial,
        temperature=temperature, max_output_tokens=max_output_tokens,
        interval_seconds=interval,
    )
    config = types.GenerateContentConfig(
        temperature=temperature, max_output_tokens=max_output_tokens,
    )
    started = perf_counter()
    try:
        response = client.models.generate_content(
            model=model, contents=question, config=config,
        )
    except errors.APIError as exc:
        result.status = "api_error"
        result.http_status = exc.code
        # 生のエラー本文やURLは保存せず、認証情報の混入を防ぐ。
        result.error = api_error_message(exc.code)
    except httpx.TransportError as exc:
        result.status = "transport_error"
        result.error = type(exc).__name__
    else:
        result.http_status = 200
    finally:
        result.elapsed_seconds = perf_counter() - started

    if result.status != "ok":
        return result
    result.model_version = response.model_version
    result.answer = response.text or ""
    if not result.answer.strip():
        result.status = "empty_response"
    if response.candidates:
        reason = response.candidates[0].finish_reason
        result.finish_reason = reason.value if reason is not None else None
    usage = response.usage_metadata
    if usage is not None:
        result.input_tokens = usage.prompt_token_count
        result.output_tokens = usage.candidates_token_count
        result.thoughts_tokens = usage.thoughts_token_count
        result.cached_tokens = usage.cached_content_token_count
        result.total_tokens = usage.total_token_count
    return result


def statistic(values: list[float | int | None], operation=mean) -> str:
    """欠測を含む平均値は表示しない（少数の既知値だけで比較しない）。"""
    if not values or any(value is None for value in values):
        return "不明"
    return f"{operation(values):.3f}"


def print_summary(results: list[Result]) -> None:
    print("\n成功した応答のみの集計（MAX_TOKENSで終了した回答も含む）")
    print("model\t質問\t成功/試行\t429\t平均秒\t中央値秒\t平均入力\t平均出力\t平均思考\t平均総数")
    pairs = dict.fromkeys((row.model, row.question) for row in results)
    for model, question in pairs:
        rows = [r for r in results if (r.model, r.question) == (model, question)]
        success = [r for r in rows if r.status == "ok"]
        elapsed = [r.elapsed_seconds for r in success]
        question_label = question.replace("\n", " ").replace("\t", " ")
        print("\t".join([
            model, question_label, f"{len(success)}/{len(rows)}",
            str(sum(r.http_status == 429 for r in rows)),
            statistic(elapsed), statistic(elapsed, median),
            *[statistic([getattr(r, field) for r in success]) for field in (
                "input_tokens", "output_tokens", "thoughts_tokens", "total_tokens",
            )],
        ]))


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("1以上の整数を指定してください。")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("0以上の有限の数を指定してください。")
    return number


def temperature_value(value: str) -> float:
    number = nonnegative_float(value)
    if number > 2:
        raise argparse.ArgumentTypeError("temperatureは0〜2で指定してください。")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=[MODEL], help="比較するモデル名")
    parser.add_argument(
        "--question", action="append", help="比較する質問（複数回指定可、省略時は3問）",
    )
    parser.add_argument("--repeat", type=positive_int, default=3, help="各組み合わせの試行回数")
    parser.add_argument(
        "--interval", type=nonnegative_float, default=15.0,
        help="前の応答が終了してから次のリクエストまでの秒数（既定15）",
    )
    parser.add_argument("--temperature", type=temperature_value, default=1.0)
    parser.add_argument("--max-output-tokens", type=positive_int, default=2048)
    parser.add_argument("--output", type=Path, help="新規CSVの保存先（上書き不可）")
    args = parser.parse_args()
    questions = list(dict.fromkeys(args.question or DEFAULT_QUESTIONS))
    models = list(dict.fromkeys(args.models))
    if any(not value.strip() for value in [*models, *questions]):
        parser.error("モデル名と質問には空文字を指定できません。")
    output = args.output or Path(
        "results/benchmark-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".csv"
    )
    api_key = get_api_key()
    output.parent.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []
    total = len(models) * len(questions) * args.repeat
    print(f"予定リクエスト数: {total} / 保存先: {output}", flush=True)
    try:
        # 自動再試行によって429や待ち時間が隠れないよう、各計測を1試行に固定。
        with output.open("x", encoding="utf-8", newline="") as stream, genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=60_000, retry_options=types.HttpRetryOptions(attempts=1),
            ),
        ) as client:
            writer = csv.DictWriter(stream, fieldnames=list(Result.__dataclass_fields__))
            writer.writeheader()
            # 反復ごと・質問ごとにモデルを交互に呼び、時間帯の偏りを軽減する。
            for trial in range(1, args.repeat + 1):
                for question in questions:
                    for model in models:
                        if results:
                            sleep(args.interval)
                        row = measure(
                            client, model, question, trial, args.temperature,
                            args.max_output_tokens, args.interval,
                        )
                        writer.writerow(asdict(row))
                        stream.flush()
                        results.append(row)
                        print(
                            f"[{len(results)}/{total}] {model} 試行{trial} "
                            f"{row.status} {row.elapsed_seconds:.3f}秒 "
                            f"入力={row.input_tokens} 出力={row.output_tokens}"
                            + (f" / {row.error}" if row.error else ""),
                            flush=True,
                        )
    except FileExistsError:
        parser.error(f"保存先が既に存在します。別のファイル名を指定してください: {output}")
    except KeyboardInterrupt:
        print("\n計測を中断しました。完了済みの結果はCSVに保存されています。")
        print_summary(results)
        raise SystemExit(130)
    print_summary(results)
    if any(row.status != "ok" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
