import pytest

from surgidoc.config import MissingKey, load_keys


def write(tmp_path, text):
    p = tmp_path / "keys.txt"
    p.write_text(text)
    return p


def test_parses_key_values_and_skips_comments_and_blanks(tmp_path):
    keys = load_keys(write(tmp_path, "# c\n\nA=1\nB = two=2 \n"))
    assert keys["A"] == "1"
    assert keys["B"] == "two=2"


def test_empty_values_are_treated_as_unset(tmp_path):
    keys = load_keys(write(tmp_path, "A=\nB=x\n"))
    assert "A" not in keys
    with pytest.raises(MissingKey, match="A"):
        keys.require("A")


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_keys(tmp_path / "nope.txt")


def test_malformed_line_rejected(tmp_path):
    with pytest.raises(ValueError, match="line 2"):
        load_keys(write(tmp_path, "A=1\nnot a pair\n"))


def test_repr_and_str_never_leak_values(tmp_path):
    keys = load_keys(write(tmp_path, "GOOGLE_AI_STUDIO_API_KEY=supersecret\n"))
    assert "supersecret" not in repr(keys)
    assert "supersecret" not in str(keys)
    assert "GOOGLE_AI_STUDIO_API_KEY" in repr(keys)


def test_environment_variables_are_not_read(tmp_path, monkeypatch):
    monkeypatch.setenv("A", "from-env")
    assert "A" not in load_keys(write(tmp_path, "B=1\n"))
