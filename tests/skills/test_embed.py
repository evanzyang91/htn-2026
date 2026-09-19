"""The local embedder, and - just as importantly - what happens when it is not there.

No test here downloads anything. The weights are a ~90 MB opt-in
(``scripts/fetch_embedder.py``), so everything that needs the model is skipped when it
is absent and everything that needs the model to be ABSENT runs everywhere. That split
is the point: the shipped default is keywords, and the fall-back is what the suite has
to keep honest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skillweaver.config import load_settings
from skillweaver.contracts import Embedder
from skillweaver.errors import ProviderError
from skillweaver.skills.embed import (
    MODEL_FILE,
    VOCAB_FILE,
    OnnxTextEmbedder,
    WordPiece,
    embedder_for,
)

# -- the tokenizer -----------------------------------------------------------------------
#
# WordPiece is written out here rather than taken from `tokenizers`, which is not a
# dependency of this project and is not going to become one: pyproject.toml is shared
# surface. So the ids below are the REFERENCE implementation's, recorded on 2026-09-19 by
# running `tokenizers.Tokenizer.from_file` over all-MiniLM-L6-v2's own tokenizer.json with
# padding and truncation off. They pinned 13 hand-picked edge cases and all five shipped
# Wikipedia skills' searchable text, byte for byte; six of the edge cases are kept here,
# chosen for what they each break if it drifts.

# fmt: off
REFERENCE_IDS = [
    # A real task sentence: the ordinary path.
    ("On the Order screen, open Copper Kettle and add a Large Pad Thai to the cart.",
     [101, 2006, 1996, 2344, 3898, 1010, 2330, 6967, 22421, 1998, 5587, 1037, 2312,
      11687, 7273, 2000, 1996, 11122, 1012, 102]),
    # A skill signature. '_' is BERT punctuation, so a snake_case name becomes the words
    # the model was trained on rather than one [UNK]: this is what makes a skill NAME
    # carry meaning at all.
    ("add_dish_with_option(restaurant: str, dish: str, option: str)",
     [101, 5587, 1035, 9841, 1035, 2007, 1035, 5724, 1006, 4825, 1024, 2358, 2099, 1010,
      9841, 1024, 2358, 2099, 1010, 5724, 1024, 2358, 2099, 1007, 102]),
    # Accents are stripped by NFD, not mapped to [UNK].
    ("Caf\u00e9 na\u00efve r\u00e9sum\u00e9", [101, 7668, 15743, 13746, 102]),
    # CJK splits per character; Cyrillic goes through WordPiece.
    ("\u0441\u043d\u0435\u0433 \u96ea \u3067\u3059",
     [101, 1196, 18947, 15290, 29741, 100, 1665, 30184, 102]),
    # Repeated punctuation stays separate tokens; an apostrophe splits the word.
    ("don't  double--dash  e-mail  C++",
     [101, 2123, 1005, 1056, 3313, 1011, 1011, 11454, 1041, 1011, 5653, 1039, 1009, 1009,
      102]),
    # An emoji is [UNK]; a zero-width space is dropped as a control character.
    ("\U0001f642 emoji and \u200b zero width",
     [101, 100, 7861, 29147, 2072, 1998, 5717, 9381, 102]),
]
# fmt: on


@pytest.fixture
def tokenizer(weights: Path) -> WordPiece:
    return WordPiece.from_file(weights / VOCAB_FILE)


@pytest.fixture
def weights() -> Path:
    """The fetched model directory, or a skip. Never a download."""
    directory = load_settings().embedder_dir
    if not (directory / MODEL_FILE).is_file() or not (directory / VOCAB_FILE).is_file():
        pytest.skip(f"no embedder weights in {directory}; run scripts/fetch_embedder.py")
    return directory


@pytest.mark.parametrize(("text", "ids"), REFERENCE_IDS, ids=lambda v: str(v)[:24])
def test_the_tokenizer_agrees_with_the_reference(
    tokenizer: WordPiece, text: str, ids: list[int]
) -> None:
    """A tokenizer that is nearly right is worse than no embedder: the vectors keep
    arriving, the ranking is quietly a little wrong, and nothing says so."""
    assert tokenizer.encode(text, max_tokens=512) == ids


def test_a_long_text_is_cut_and_still_closed(tokenizer: WordPiece) -> None:
    ids = tokenizer.encode("order the pad thai " * 200, max_tokens=32)
    assert len(ids) == 32
    assert ids[0] == tokenizer.cls_id and ids[-1] == tokenizer.sep_id


def test_an_empty_text_is_two_special_tokens(tokenizer: WordPiece) -> None:
    assert tokenizer.encode("") == [tokenizer.cls_id, tokenizer.sep_id]
    assert tokenizer.encode("   \t\n ") == [tokenizer.cls_id, tokenizer.sep_id]


def test_a_vocabulary_without_the_special_tokens_is_refused() -> None:
    with pytest.raises(ProviderError, match=r"\[CLS\]"):
        WordPiece({"[UNK]": 0, "[SEP]": 1, "hello": 2})


def test_a_missing_vocabulary_file_is_a_provider_error(tmp_path: Path) -> None:
    with pytest.raises(ProviderError, match="cannot read"):
        WordPiece.from_file(tmp_path / "nope.txt")


# -- the contract ------------------------------------------------------------------------


@pytest.fixture
def embedder(weights: Path) -> OnnxTextEmbedder:
    return OnnxTextEmbedder(weights / MODEL_FILE, weights / VOCAB_FILE)


def test_it_satisfies_the_embedder_protocol(embedder: OnnxTextEmbedder) -> None:
    assert isinstance(embedder, Embedder)


def test_empty_input_gives_empty_output_and_loads_nothing(weights: Path) -> None:
    """The contract's own corner, and worth its own test because it is the one call
    that must not pay for a 90 MB session."""
    fresh = OnnxTextEmbedder(weights / MODEL_FILE, weights / VOCAB_FILE)
    assert fresh.embed([]) == []
    assert "not loaded yet" in repr(fresh)


def test_every_vector_is_the_same_length_and_l2_normalized(
    embedder: OnnxTextEmbedder,
) -> None:
    """``SkillRetriever`` takes a dot product and calls it a cosine. This is what
    entitles it to."""
    vectors = embedder.embed(["pay a bill", "", "settle an invoice at once"])
    assert len(vectors) == 3
    assert {len(v) for v in vectors} == {384}
    for vector in vectors:
        assert sum(v * v for v in vector) == pytest.approx(1.0, abs=1e-5)


def test_the_same_text_gives_the_same_vector(embedder: OnnxTextEmbedder) -> None:
    """Deterministic, as the protocol says - and across batch shapes, because the
    retriever caches a vector computed in one batch and reuses it in another."""
    alone = embedder.embed(["settle an invoice"])[0]
    in_company = embedder.embed(["pay a bill", "settle an invoice", "x"])[1]
    assert alone == pytest.approx(in_company, abs=1e-5)


def test_it_knows_that_two_wordings_are_one_errand(embedder: OnnxTextEmbedder) -> None:
    """The whole reason to run a model at all: no word of "pay a bill" appears in
    "settle an invoice", and token overlap therefore scores them zero."""
    bill, invoice, telescope = embedder.embed(
        ["pay a bill", "settle an invoice", "recalibrate the telescope mirror"]
    )
    same = sum(a * b for a, b in zip(bill, invoice, strict=True))
    unrelated = sum(a * b for a, b in zip(bill, telescope, strict=True))
    assert same > 0.4 > unrelated


def test_a_model_that_will_not_load_is_a_provider_error(tmp_path: Path, weights: Path) -> None:
    (tmp_path / MODEL_FILE).write_bytes(b"not an onnx graph")
    broken = OnnxTextEmbedder(tmp_path / MODEL_FILE, weights / VOCAB_FILE)
    with pytest.raises(ProviderError, match="cannot load embedding model"):
        broken.embed(["anything"])


# -- whether there is one at all ---------------------------------------------------------
#
# These run on every machine, weights or not: they are the shipped default.


def settings_and_embedder(tmp_path: Path, **env: str) -> tuple[OnnxTextEmbedder | None, str]:
    """:func:`embedder_for` over an environment that names ``tmp_path`` as the place
    the weights would be, so these tests never see the real ones."""
    return embedder_for(
        load_settings({"SKILLWEAVER_EMBEDDER_DIR": str(tmp_path), **env}, env_file=None)
    )


def test_no_weights_means_no_embedder_and_a_reason_that_names_the_fix(tmp_path: Path) -> None:
    embedder, reason = settings_and_embedder(tmp_path, SKILLWEAVER_EMBEDDER="true")
    assert embedder is None
    assert "no local weights" in reason and "fetch_embedder" in reason


def test_half_the_files_is_still_no_embedder(tmp_path: Path) -> None:
    (tmp_path / VOCAB_FILE).write_text("[UNK]\n[CLS]\n[SEP]\n", encoding="utf-8")
    embedder, reason = settings_and_embedder(tmp_path, SKILLWEAVER_EMBEDDER="true")
    assert embedder is None
    assert MODEL_FILE in reason and VOCAB_FILE not in reason


def test_turning_it_off_says_so_rather_than_going_quiet(tmp_path: Path) -> None:
    """The reason travels into every candidate's ``why``, so a run ranked on keywords
    can be told apart from a run that ranked on keywords BECAUSE somebody turned the
    model off. A measurement that cannot tell those apart is not a measurement."""
    embedder, reason = settings_and_embedder(tmp_path, SKILLWEAVER_EMBEDDER="false")
    assert embedder is None
    assert "turned off by SKILLWEAVER_EMBEDDER" in reason


def test_off_is_the_shipped_default(tmp_path: Path) -> None:
    """Pinned because it is a measured decision, not an oversight: ranking by meaning
    improves recall and moves no run. See DEFAULT_EMBEDDER_ENABLED, and re-measure
    with scripts/bench_retrieval.py before changing this."""
    embedder, reason = settings_and_embedder(tmp_path)
    assert embedder is None
    assert "turned off" in reason


def test_weights_on_disk_do_not_switch_it_on_by_themselves(weights: Path) -> None:
    """The one way to enable this is the explicit switch, and fetching the model is
    NOT that way.

    It matters because the weights are ~90 MB that somebody downloads once and then
    forgets about, possibly for the bench and possibly for a different worktree
    sharing the same directory. If their presence enabled ranking by meaning, a
    machine that had run `make embedder` months ago would quietly rank differently
    from a clean clone - and the measurement says that ranking answers five of six
    irrelevant requests instead of one. So the check order in `embedder_for` is
    load-bearing: the switch is read BEFORE the disk is looked at.
    """
    config = load_settings({"SKILLWEAVER_EMBEDDER_DIR": str(weights)}, env_file=None)
    embedder, reason = embedder_for(config)

    assert embedder is None, "present weights must not be an implicit yes"
    assert "turned off" in reason


def test_naming_the_weights_directory_is_not_a_way_to_turn_it_on(weights: Path) -> None:
    """SKILLWEAVER_EMBEDDER_DIR exists so several worktrees can share one download.
    Setting it for that purpose must not also flip the ranking."""
    config = load_settings(
        {"SKILLWEAVER_EMBEDDER_DIR": str(weights), "SKILLWEAVER_EMBEDDER": "false"},
        env_file=None,
    )
    assert embedder_for(config)[0] is None


def test_weights_present_and_asked_for_gives_one(weights: Path) -> None:
    config = load_settings(
        {"SKILLWEAVER_EMBEDDER": "true", "SKILLWEAVER_EMBEDDER_DIR": str(weights)},
        env_file=None,
    )
    embedder, reason = embedder_for(config)
    assert isinstance(embedder, OnnxTextEmbedder)
    assert reason == ""


def test_the_fetch_script_pins_the_digests_it_verifies() -> None:
    """The one network step this feature has, and the one thing that makes it safe:
    a truncated download has to fail loudly rather than become a ranking that is
    quietly wrong. Read as text so importing the script - and with it urllib - is not
    part of the suite."""
    source = (Path(__file__).resolve().parents[2] / "scripts" / "fetch_embedder.py").read_text()
    assert "sha256" in source
    assert source.count('"6fd5d72f') == 1, "the model digest is pinned exactly once"
    assert "all-MiniLM-L6-v2" in source
