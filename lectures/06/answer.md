# 第6回 解答

## 演習

<!-- [tool call] の行は、表示された行をそのまま貼り付けてください。終わらなかった STEP は、空欄のままでかまいません。 -->

- STEP 1 のツールの名前：search_glossary
- STEP 2 の [tool call]：
    [tool call]   glossary_search_glossary({'query': '3レベル'})
- STEP 3 の [tool call]：
    [tool call]   fetch_fetch({'url': 'https://modelcontextprotocol.io/'})
- STEP 4 の [tool call]：
    [tool call]   aws_aws___search_documentation({'topics': ['general'], 'search_phrase': 'Amazon S3 versioning overview'})
- 気づき・疑問：
    用語集と fetch はコマンドで起動し、AWS は URL で接続した。どのツールも `--trace` で呼び出し時の引数と戻り値を確認できた。AWS の例では、日本語の質問からモデルが `Amazon S3 versioning overview` という英語の検索語を作って渡していた。

## 宿題

<!-- 基礎・実装・発展のどれか1つに取り組めば完了です。 -->

- 取り組んだレベル：基礎
- やったこと：`app/mcp_server.py` に `server.tool()(tools.get_lecture_summary)` を追加し、各回の要点を取得する関数を MCP ツールとして公開した。

    ```text
    $ uv run python -m app.mcp_agent --list-tools
    [glossary_get_lecture_summary]

    $ uv run python -m app.mcp_agent --trace "第2回では何を学びますか？"
    [tool call]   glossary_get_lecture_summary({'lecture': 2})
    [tool return] glossary_get_lecture_summary -> # 第2回 ファインチューニングとLoRA

    $ uv run pytest -q
    21 passed in 4.84s
    ```

- 結果：`glossary_get_lecture_summary` がツール一覧に表示された。第2回について質問すると、モデルはツールが返した要点を使って回答した。テストは21件すべて通った。

    ```
    第2回では、「**ファインチューニングとLoRA**」について学びます。

    [usage] 入力 5,942 tokens / 出力 226 tokens / モデル応答 2 回
    ```

    ```
    $ uv run pytest -q
    .....................                                                    [100%]
    21 passed in 4.84s
    ```
- 気づき・疑問：エージェントは起動時に MCP サーバーからツール一覧を取得するため、サーバー側に登録を1行追加するだけで、新しいツールを一覧表示して質問への回答に使えることが分かった。
