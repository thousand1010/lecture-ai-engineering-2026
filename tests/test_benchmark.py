import csv
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors, types

from app import benchmark


def response(usage=True, text="回答", reason=types.FinishReason.STOP):
    return types.GenerateContentResponse(
        model_version="actual-model-version",
        candidates=[types.Candidate(
            content=types.Content(role="model", parts=[types.Part(text=text)]),
            finish_reason=reason,
        )],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=12, candidates_token_count=8,
            thoughts_token_count=4, total_token_count=24,
            cached_content_token_count=0,
        ) if usage else None,
    )


@pytest.fixture
def fake_client(monkeypatch):
    requests = []
    outcomes = []
    options = []

    def generate(**kwargs):
        requests.append(kwargs)
        outcome = outcomes.pop(0) if outcomes else response()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))

    class Context:
        def __init__(self, **kwargs):
            options.append(kwargs)

        def __enter__(self):
            return client

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(benchmark.genai, "Client", Context)
    monkeypatch.setattr(benchmark, "get_api_key", lambda: "test-key")
    return client, requests, outcomes, options


def test_measure_records_timing_usage_and_generation_settings(monkeypatch, fake_client):
    client, requests, _, _ = fake_client
    times = iter([100.0, 102.25])
    monkeypatch.setattr(benchmark, "perf_counter", lambda: next(times))

    result = benchmark.measure(client, "requested-model", "質問", 2, 0.5, 512, 15)

    assert result.elapsed_seconds == 2.25
    assert (result.input_tokens, result.output_tokens, result.thoughts_tokens) == (12, 8, 4)
    assert result.total_tokens == 24  # 思考トークンも含み、入力+出力とは異なる。
    assert result.cached_tokens == 0
    assert result.model_version == "actual-model-version"
    assert result.finish_reason == "STOP"
    assert result.answer == "回答"
    assert result.status == "ok" and result.http_status == 200
    assert requests[0]["contents"] == "質問"
    assert requests[0]["config"].temperature == 0.5
    assert requests[0]["config"].max_output_tokens == 512


def test_missing_usage_and_empty_answer_are_not_zero(fake_client):
    client, _, outcomes, _ = fake_client
    outcomes.append(response(usage=False, text=""))

    result = benchmark.measure(client, "model", "質問", 1, 1, 512, 0)

    assert result.status == "empty_response"
    assert result.input_tokens is None and result.output_tokens is None


@pytest.mark.parametrize("failure,status,code", [
    (errors.APIError(429, {"message": "secret"}), "api_error", 429),
    (errors.APIError(404, {"message": "secret"}), "api_error", 404),
    (httpx.ReadTimeout("secret"), "transport_error", None),
])
def test_failures_record_time_and_do_not_leak_raw_details(
    monkeypatch, fake_client, failure, status, code,
):
    client, _, outcomes, _ = fake_client
    outcomes.append(failure)
    times = iter([1.0, 4.0])
    monkeypatch.setattr(benchmark, "perf_counter", lambda: next(times))

    result = benchmark.measure(client, "model", "質問", 1, 1, 512, 0)

    assert result.status == status and result.http_status == code
    assert result.elapsed_seconds == 3.0
    assert result.total_tokens is None
    assert "secret" not in result.error


def test_cli_compares_independent_requests_and_saves_errors(
    monkeypatch, tmp_path, capsys, fake_client,
):
    _, requests, outcomes, options = fake_client
    outcomes.extend([errors.APIError(429, {"message": "quota"}), response(usage=False)])
    target = tmp_path / "results.csv"
    monkeypatch.setattr("sys.argv", [
        "benchmark", "--models", "m1", "m2", "--question", "q1", "--question", "q2",
        "--repeat", "2", "--interval", "12", "--output", str(target),
    ])
    waits = []

    def wait(seconds):
        # 次の呼び出し前に前回の結果が既に保存されている。
        with target.open() as stream:
            assert len(list(csv.DictReader(stream))) == len(requests)
        waits.append(seconds)

    monkeypatch.setattr(benchmark, "sleep", wait)
    with pytest.raises(SystemExit) as exc:
        benchmark.main()

    assert exc.value.code == 1
    assert [(r["model"], r["contents"]) for r in requests] == [
        ("m1", "q1"), ("m2", "q1"), ("m1", "q2"), ("m2", "q2"),
    ] * 2
    assert waits == [12] * 7
    assert options[0]["http_options"].retry_options.attempts == 1
    assert options[0]["http_options"].timeout == 60_000
    with target.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert rows[0]["http_status"] == "429" and rows[0]["input_tokens"] == ""
    assert rows[1]["status"] == "ok" and rows[1]["input_tokens"] == ""
    assert rows[-1]["trial"] == "2"
    output = capsys.readouterr().out
    assert "m1\tq1\t1/2\t1" in output
    assert "m2\tq1\t2/2\t0" in output
    assert "不明" in output


def test_existing_csv_is_preserved(monkeypatch, tmp_path, fake_client):
    target = tmp_path / "results.csv"
    target.write_text("previous data")
    monkeypatch.setattr("sys.argv", ["benchmark", "--output", str(target)])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
    assert target.read_text() == "previous data"
    assert fake_client[1] == []


def test_interrupt_keeps_completed_rows(monkeypatch, tmp_path, fake_client):
    target = tmp_path / "results.csv"
    monkeypatch.setattr("sys.argv", [
        "benchmark", "--question", "q", "--repeat", "2", "--output", str(target),
    ])

    def interrupt(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(benchmark, "sleep", interrupt)
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 130
    with target.open() as stream:
        assert len(list(csv.DictReader(stream))) == 1


@pytest.mark.parametrize("args", [
    ["--repeat", "0"], ["--interval", "-1"], ["--interval", "nan"],
    ["--temperature", "inf"], ["--temperature", "2.1"],
    ["--max-output-tokens", "0"], ["--question", " "], ["--models", ""],
])
def test_invalid_settings_do_not_call_api(monkeypatch, args, fake_client):
    monkeypatch.setattr("sys.argv", ["benchmark", *args])
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
    assert fake_client[1] == []
