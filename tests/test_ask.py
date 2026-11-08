from types import SimpleNamespace

import pytest
from google.genai import types

from app import ask


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setattr(ask, "perf_counter", lambda: 0.0)


def test_ask_prints_input_and_output_token_counts(monkeypatch, capsys):
    times = iter([10.0, 12.5])
    monkeypatch.setattr(ask, "perf_counter", lambda: next(times))
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=8),
        text="回答",
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    assert ask.ask("質問") == "回答"
    assert capsys.readouterr().out == (
        "[今回の入力トークン数] 12\n[今回の出力トークン数] 8\n"
        "[セッション累計の入力トークン数] 12\n[セッション累計の出力トークン数] 8\n"
        "[応答時間（秒）] 2.500\n"
    )


def test_ask_reports_unavailable_token_counts(monkeypatch, capsys):
    response = SimpleNamespace(usage_metadata=None, text="回答")
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    ask.ask("質問")

    assert capsys.readouterr().out == (
        "[今回の入力トークン数] 不明\n[今回の出力トークン数] 不明\n"
        "[セッション累計の入力トークン数] 不明\n[セッション累計の出力トークン数] 不明\n"
        "[応答時間（秒）] 0.000\n"
    )


@pytest.fixture
def conversation_client(monkeypatch):
    requests = []
    replies = []

    def generate_content(**kwargs):
        requests.append(kwargs)
        content = types.Content(
            role="model",
            parts=[types.Part(text=f"回答{len(requests)}", thought_signature=b"signature")],
        )
        replies.append(content)
        return types.GenerateContentResponse(
            candidates=[types.Candidate(content=content)],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=10 * len(requests),
                candidates_token_count=5 * len(requests),
            ),
        )

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    return requests, replies


def test_ask_preserves_questions_and_full_model_replies(conversation_client):
    requests, replies = conversation_client
    history = []

    assert ask.ask("LoRAとは？", history) == "回答1"
    assert ask.ask("それを短く説明して", history) == "回答2"

    assert [part.role for part in history] == ["user", "model", "user", "model"]
    assert len(requests[0]["contents"]) == 1
    assert requests[1]["model"] == ask.MODEL
    assert [part.parts[0].text for part in requests[1]["contents"]] == [
        "LoRAとは？", "回答1", "それを短く説明して"
    ]
    assert requests[1]["contents"][1] is replies[0]

    ask.ask("別の会話", [])
    assert len(requests[2]["contents"]) == 1


def test_ask_leaves_history_unchanged_on_api_error(monkeypatch, conversation_client):
    history = []
    totals = ask.TokenTotals()
    ask.ask("最初の質問", history, totals)
    previous = history.copy()

    def fail(**kwargs):
        raise ask.errors.APIError(429, {"message": "rate limit"})

    monkeypatch.setattr(
        ask.genai, "Client",
        lambda **kwargs: SimpleNamespace(models=SimpleNamespace(generate_content=fail)),
    )
    with pytest.raises(ask.errors.APIError):
        ask.ask("失敗する質問", history, totals)
    assert history == previous
    assert totals == ask.TokenTotals(input_tokens=10, output_tokens=5)


@pytest.mark.parametrize("candidates", [None, [types.Candidate()]])
def test_ask_handles_missing_answer(monkeypatch, candidates):
    client = SimpleNamespace(models=SimpleNamespace(
        generate_content=lambda **kwargs: types.GenerateContentResponse(candidates=candidates)
    ))
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    history = []

    assert ask.ask("質問", history) == "回答が生成されませんでした。"
    assert history == []


@pytest.mark.parametrize("ending", ["exit", "QUIT", EOFError, KeyboardInterrupt])
@pytest.mark.parametrize("initial", [[], ["最初の質問"]])
def test_main_continues_conversation_and_exits(
    monkeypatch, capsys, conversation_client, ending, initial
):
    requests, _ = conversation_client
    monkeypatch.setattr("sys.argv", ["ask", *initial])
    inputs = iter(["  ", "続きの質問", ending])

    def read_input(prompt):
        value = next(inputs)
        if isinstance(value, type):
            raise value()
        return value

    monkeypatch.setattr("builtins.input", read_input)
    ask.main()

    assert len(requests) == 2
    assert requests[1]["contents"][0].parts[0].text == (
        initial[0] if initial else ask.DEFAULT_QUESTION
    )
    assert requests[1]["contents"][1].parts[0].text == "回答1"
    assert requests[1]["contents"][2].parts[0].text == "続きの質問"
    output = capsys.readouterr().out
    assert "回答1" in output and "回答2" in output
    assert "[今回の入力トークン数] 20" in output
    assert "[今回の出力トークン数] 10" in output
    assert "[セッション累計の入力トークン数] 30" in output
    assert "[セッション累計の出力トークン数] 15" in output


@pytest.mark.parametrize("missing_input", [True, False])
def test_unknown_usage_keeps_affected_total_unknown(
    monkeypatch, capsys, conversation_client, missing_input
):
    totals = ask.TokenTotals(input_tokens=10, output_tokens=5)
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=None if missing_input else 0,
            candidates_token_count=0 if missing_input else None,
        ),
        text="回答",
    )
    with monkeypatch.context() as patch:
        patch.setattr(ask.genai, "Client", lambda **kwargs: SimpleNamespace(
            models=SimpleNamespace(generate_content=lambda **kwargs: response)
        ))
        ask.ask("質問", totals=totals)
    capsys.readouterr()

    ask.ask("続き", totals=totals)
    output = capsys.readouterr().out
    if missing_input:
        assert "[今回の入力トークン数] 10" in output
        assert "[セッション累計の入力トークン数] 不明" in output
        assert totals == ask.TokenTotals(input_tokens=None, output_tokens=10)
    else:
        assert "[今回の出力トークン数] 5" in output
        assert "[セッション累計の出力トークン数] 不明" in output
        assert totals == ask.TokenTotals(input_tokens=20, output_tokens=None)


def test_main_resets_totals_for_each_session(monkeypatch, capsys, conversation_client):
    monkeypatch.setattr("sys.argv", ["ask", "質問"])
    monkeypatch.setattr("builtins.input", lambda prompt: "exit")
    ask.main()
    capsys.readouterr()

    ask.main()

    output = capsys.readouterr().out
    assert "[セッション累計の入力トークン数] 20" in output
    assert "[セッション累計の出力トークン数] 10" in output
