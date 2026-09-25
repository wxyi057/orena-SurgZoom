from surgzoom.prompt import build_system_prompt, user_content
from surgzoom.zoom import normalize_answer, window_around, zoom_anchor


def test_zoom_only_for_times_inside_long_clips():
    assert zoom_anchor("00:51:20", 3000, 3119) == 3080
    assert zoom_anchor("00:00:22", 3000, 3119) is None      # a duration, not a time
    assert zoom_anchor("3", 3000, 3119) is None
    assert zoom_anchor("00:51:20", 3000, 3029) is None      # clip shorter than 60 s
    assert zoom_anchor("<think>x</think>00:49:40", 3000, 3119) == 3000   # clamped into the clip


def test_window_geometry_shifts_inwards_at_edges():
    assert window_around(3060, 15, 3000, 3119) == (3045, 3075)
    assert window_around(3005, 15, 3000, 3119) == (3000, 3030)
    assert window_around(3115, 7.5, 3000, 3119) == (3104, 3119)


def test_normalisation():
    assert normalize_answer("00:31:33, 00:32:17") == "00:31:33"
    assert normalize_answer("00:31:33 and 00:32:17.") == "00:31:33"
    assert normalize_answer("The needle is removed at 00:31:33.") == "The needle is removed at 00:31:33."
    assert normalize_answer("<think>hmm</think>Sponge") == "Sponge"


def test_prompt_format():
    assert build_system_prompt("DEFS").endswith("Be precise and concise.\n\nDEFS")
    assert user_content("Q?", 3000, 3119, "sigmoid resection") == (
        "<video>Procedure type: sigmoid resection.\n"
        "Clip window: 00:50:00 - 00:51:59 (source-video timeline).\nQ?")
    assert user_content("Q?", 0, 29).startswith("<video>Clip window: 00:00:00 - 00:00:29")
