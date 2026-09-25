from surgzoom.policy import frames_for, route, sampling_kwargs


def test_time_questions_get_dense_sampling():
    q = "When is the Sponge first visible? Please provide the answer in the format hh:mm:ss."
    assert route(q) == "time" and frames_for(q) == 256


def test_counting_questions_get_dense_sampling():
    assert route("How many Clip(s) are inserted in the abdomen in this video? Please provide a number.") == "count"
    assert route("What is the maximum number of Clips appearing at once in a single frame?") == "count"
    q = ("After the Specimen Bag was inserted, which other foreign object classes "
         "are visible in this video?")
    assert route(q) == "count"


def test_single_frame_and_percentage_questions_are_not_counting():
    assert route("How many Clips are visible in frame 00:31:12?") == "other"
    assert route("In %, how many of the frames show a Sponge?") == "other"


def test_other_questions_get_resolution():
    q = "Which surgical foreign object is inserted in this video? Please provide a class name."
    assert route(q) == "other" and frames_for(q) == 128
    assert sampling_kwargs(q) == {"min_frames": 128, "max_frames": 128}


def test_loose_time_rule_only_at_inference():
    q = "At what time does the needle leave the field of view?"
    assert route(q, loose=True) == "time"
    assert route(q, loose=False) == "other"
