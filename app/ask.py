"""第1回: 素のLLMに会話履歴を引き継いで質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

回答の後に続けて質問できる。exit / quit、EOF、Ctrl+C で終了する。
会話履歴は実行中だけ保持し、終了すると消える。
各回答で今回とセッション累計の入力・出力トークン数を表示する。
API 呼び出し開始から応答全体の受信までの時間も表示する。
ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
from dataclasses import dataclass
from time import perf_counter

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


@dataclass
class TokenTotals:
    """セッションの累計。取得できない値があれば、その累計も不明にする。"""

    input_tokens: int | None = 0
    output_tokens: int | None = 0

    def add(self, input_tokens: int | None, output_tokens: int | None) -> None:
        self.input_tokens = (
            self.input_tokens + input_tokens
            if self.input_tokens is not None and input_tokens is not None else None
        )
        self.output_tokens = (
            self.output_tokens + output_tokens
            if self.output_tokens is not None and output_tokens is not None else None
        )


def ask(
    question: str,
    history: list[types.Content] | None = None,
    totals: TokenTotals | None = None,
) -> str:
    """質問に答え、history を渡した場合は成功したやり取りを追記する。"""
    client = genai.Client(api_key=get_api_key())
    user_content = types.Content(role="user", parts=[types.Part(text=question)])
    contents = [*history, user_content] if history is not None else question
    started = perf_counter()
    response = client.models.generate_content(model=MODEL, contents=contents)
    elapsed = perf_counter() - started

    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage is not None else None
    output_tokens = usage.candidates_token_count if usage is not None else None
    if totals is None:
        totals = TokenTotals()
    # 入力は会話履歴の再送分も含め、API が返した使用量を毎回加算する。
    totals.add(input_tokens, output_tokens)
    for label, count in (
        ("今回の入力トークン数", input_tokens),
        ("今回の出力トークン数", output_tokens),
        ("セッション累計の入力トークン数", totals.input_tokens),
        ("セッション累計の出力トークン数", totals.output_tokens),
    ):
        print(f"[{label}] {count if count is not None else '不明'}")
    print(f"[応答時間（秒）] {elapsed:.3f}")

    # 回答のパーツをそのまま保持し、モデルが返した付加情報も引き継ぐ。
    if history is not None and response.candidates:
        model_content = response.candidates[0].content
        if model_content is not None and model_content.parts:
            history.extend([user_content, model_content])

    return response.text or "回答が生成されませんでした。"


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに会話履歴を引き継いで質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print("続けて質問できます。exit / quit、EOF、Ctrl+C で終了します。")
    history: list[types.Content] = []
    totals = TokenTotals()
    question = args.question
    try:
        while True:
            print(f"[質問] {question}\n")
            print(ask(question, history, totals))
            while True:
                question = input("\n[次の質問] ").strip()
                if question.lower() in ("exit", "quit"):
                    return
                if question:
                    break
    except (EOFError, KeyboardInterrupt):
        print("\n会話を終了します。")
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
