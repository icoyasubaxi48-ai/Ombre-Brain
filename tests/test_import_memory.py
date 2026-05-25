from import_memory import chunk_turns, detect_and_parse
from utils import count_tokens_approx


def test_markdown_parser_supports_chinese_role_prefixes():
    text = "用户：这里有 <b>HTML</b> & 符号\n助手：收到，我不会改写原文。"

    turns = detect_and_parse(text, "chat.md")

    assert [(turn["role"], turn["content"]) for turn in turns] == [
        ("user", "这里有 <b>HTML</b> & 符号"),
        ("assistant", "收到，我不会改写原文。"),
    ]


def test_markdown_parser_supports_ascii_role_prefixes():
    text = "user: first\nassistant: second\nHuman: third\n**AI:** fourth"

    turns = detect_and_parse(text, "chat.md")

    assert [(turn["role"], turn["content"]) for turn in turns] == [
        ("user", "first"),
        ("assistant", "second"),
        ("user", "third"),
        ("assistant", "fourth"),
    ]


def test_oversized_markdown_turn_is_split_into_multiple_chunks():
    long_line = "这是一段很长的导入文本。" * 700
    turns = detect_and_parse("用户：" + long_line, "chat.md")

    chunks = chunk_turns(turns, target_tokens=800)

    assert len(chunks) > 1
    assert all(count_tokens_approx(chunk["content"]) <= 1200 for chunk in chunks)
    assert all(chunk["content"].startswith("[用户] ") for chunk in chunks)
