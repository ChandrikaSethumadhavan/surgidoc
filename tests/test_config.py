import pytest

from surgidoc.config import MissingKey, load_keys


def write(tmp_path, text):  #It lets each test provide its own keys file without touching real files, so tests are isolated and repeatable.
    p = tmp_path / "keys.txt"
    p.write_text(text)
    return p


def test_parses_key_values_and_skips_comments_and_blanks(tmp_path):
    keys = load_keys(write(tmp_path, "# c\n\nA=1\nB = two=2 \n"))
    assert keys["A"] == "1"
    assert keys["B"] == "two=2"
    
    #"# c\n\nA=1\nB = two=2 \n" is one string containing the sample file’s contents. \n means “new line,” so it represents: write(...) saves that text to a temporary keys.txt; load_keys(...) reads and parses it. The two asserts check that the parser skips the comment and blank line, 
#trims whitespace, and reads A as "1" and B as "two=2".


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
    
    #This test checks that displaying keys doesn’t reveal secret values:
# It writes a sample key/value into a temporary file and loads it.
# repr(keys) is Python’s developer-facing representation; str(keys) is its regular string form.
# The first two assertions verify neither display includes "supersecret".
# The last assertion verifies the key name is still shown, so you can identify which keys are loaded without exposing their values.


def test_environment_variables_are_not_read(tmp_path, monkeypatch):
    monkeypatch.setenv("A", "from-env")
    assert "A" not in load_keys(write(tmp_path, "B=1\n"))


# This test verifies that load_keys reads keys from the file, not from environment variables.

# monkeypatch.setenv("A", "from-env") temporarily sets environment variable A for this test.
# write(tmp_path, "B=1\n") creates a temporary keys file containing only B=1.
# load_keys(...) loads that file.
# The assertion checks that "A" is not among the loaded keys, even though it exists in the environment.
# Pytest’s monkeypatch restores the environment after the test, so it won’t affect other tests.