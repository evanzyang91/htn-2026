# Detector test fixtures

Committed screenshots and their DOM ground truth, used by
`tests/perception/test_detect_yolo.py` to score the trained detector with no
network and no training step.

They come in two groups, and the test never averages them together.

**`sandbox`** - three frames of the sandbox app at a viewport (1200x780) that is in
neither split of the training dataset, one of them at device scale 2.0, which is
what keeps the test honest about the physical-to-logical conversion.

**`web`** - four photographs of live public pages, none of whose URLs are in the
training set, and two of whose SITES are not in it at all:

| fixture | page | seen in training? |
| --- | --- | --- |
| `wiki_article` | en.wikipedia.org article | the site, not this page |
| `wiki_results` | en.wikipedia.org search results | the site, not this query |
| `sqlite_home` | sqlite.org | no - site never trained on |
| `python_org` | python.org | no - site never trained on |

Labels are ordinary YOLO label files: class id from
`skillweaver.perception.labeling.CLASS_NAMES`, then `cx cy w h` normalized against
the LOGICAL frame size in `frames.json`.

## What the shipped weights measure here

Recall at 0.5 IoU, matching element kind, against the DOM ground truth captured
with each frame. `old` is the first version of these weights, trained on 240 frames
of the sandbox app and nothing else. `new` is what ships now: 425 frames of the
sandbox app, 12 live Wikipedia pages and 6 pages from 5 other sites.

| frame | group | elements | old | new |
| --- | --- | --- | --- | --- |
| `mail_inbox` (1x) | sandbox | 57 | 1.000 | 1.000 |
| `rec_selected` (1x) | sandbox | 185 | 0.941 | 0.957 |
| `set_dialog` (2x) | sandbox | 48 | 0.792 | 0.833 |
| `wiki_article` | web | 102 | 0.039 | 0.618 |
| `wiki_results` | web | 71 | 0.099 | 0.718 |
| `sqlite_home` | web | 61 | 0.016 | 0.377 |
| `python_org` | web | 63 | 0.032 | 0.095 |
| **sandbox, mean** | | | **0.911** | **0.930** |
| **web, mean** | | | **0.046** | **0.452** |
| **sandbox, interactive kinds pooled** | | 95 | **0.958** | **0.948** |
| **web, interactive kinds pooled** | | 180 | **0.028** | **0.583** |

The same frames by the two numbers the gap was originally reported in - mean
confidence over every detection returned, and how many elements are found per
frame:

| | old | new |
| --- | --- | --- |
| sandbox, mean confidence | 0.886 | 0.892 |
| sandbox, elements per frame | 91.7 | 92.7 |
| real pages, mean confidence | 0.499 | 0.648 |
| real pages, elements per frame | 30.5 | 54.0 |

## What is still wrong, stated plainly

**A dark page is close to a blind screen.** `python_org` is light text on a dark
blue field, and it goes from 0.032 to 0.095 - a threefold improvement of a number
that is still almost zero. Nearly every frame in the training set is dark-on-light,
so the detector has barely seen the other polarity. Frames from dark-themed pages,
or a colour-inversion augmentation over the frames already harvested, is the
obvious next thing to try and has not been tried.

**Hacker News is a total blind spot.** It is not a fixture because there is nothing
to pin: on its front page the detector finds 11 boxes where the DOM reports 309,
for a recall of 0.000, and that is true of all three sets of weights. Eight-point
text in a 2007 table layout is further from anything in the training set than any
amount of extra Wikipedia was going to fix. `--bench` keeps measuring it anyway.

**One site is not the web.** Six real sites is enough to move an unseen ordinary
site from 0.016 to 0.377 (`sqlite_home`) and from 0.000 to 0.559 (postgresql.org,
measured by `--bench`), which is real generalization and is also nowhere near
solved. The honest summary is: very good on the app it was trained on, good on
pages of a site it was trained on, partial on an unseen ordinary site, blind on
dark themes and on micro-dense layouts.

## Regenerating

    uv run python scripts/build_ui_dataset.py --fixtures tests/perception/fixtures

The sandbox frames are deterministic and come back byte-identical unless the app
itself changed. The web frames are photographs of live pages and will NOT: they
need the network, and the pinned floors have to be remeasured after a regeneration
rather than adjusted until the test passes.

To remeasure everything, including the pages too unstable or too large to commit:

    uv run python scripts/build_ui_dataset.py --bench \
        --weights old.pt --weights data/models/ui_detector.pt
