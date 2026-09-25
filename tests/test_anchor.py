import pytest

from surgzoom.anchor import anchor_family, anchor_window

S, E = 3000.0, 3119.0      # a 119 s clip: 00:50:00 - 00:51:59


@pytest.mark.parametrize("question,family,window", [
    ("There is one Sponge in the frame at 00:50:40. When is it retrieved from the surgical site?",
     "retrieval", (3030.0, 3119.0)),
    ("Does the Needle, last visible just before 00:51:00, re-appear later in this video?",
     "reappear", (3030.0, 3119.0)),
    ("Is a Clip seen between 00:50:30 and 00:50:40?", "range", (3025.0, 3045.0)),
    ("Does the Sponge at 00:50:20 also appear at 00:51:00?", "cooccur", (3010.0, 3070.0)),
    ("Is the Needle visible at 00:50:50 being retrieved at that moment?", "moment", (3040.0, 3060.0)),
    ("What foreign object was inserted or created at 00:50:50?", "insert", (3035.0, 3065.0)),
    ("In which quadrant does the Sponge first occur in this video at 00:50:20?", "quadrant", (3000.0, 3030.0)),
    ("How many Clips are visible in frame 00:51:30?", "frame_local", (3080.0, 3100.0)),
])
def test_families_and_windows(question, family, window):
    assert anchor_family(question) == family
    assert anchor_window(question, S, E) == window


def test_no_time_means_whole_clip():
    assert anchor_family("How many Clips are in this video?") is None
    assert anchor_window("How many Clips are in this video?", S, E) is None


def test_time_outside_clip_means_whole_clip():
    assert anchor_window("How many Clips are visible in frame 01:00:00?", S, E) is None


def test_window_covering_almost_the_clip_is_not_cut():
    # retrieval window [t-10, end] on a short clip covers > 95 % of it
    assert anchor_window("There is one Sponge in the frame at 00:50:05. When is it retrieved?",
                         3000.0, 3029.0) is None


def test_minimum_window_at_clip_edge():
    w = anchor_window("How many Clips are visible in frame 00:51:58?", S, E)
    assert w == (3099.0, 3119.0)
