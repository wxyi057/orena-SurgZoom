# Examples

`python tools/make_examples.py` builds `examples/demo/` from the official
[HeiCo-FOCUS-VQA](https://huggingface.co/datasets/orena-dkfz/heico-focus-vqa) and
[LapChole-FOCUS-VQA](https://huggingface.co/datasets/orena-dkfz/lapchole-focus-vqa) releases, in the
challenge input layout. Request access on the Hub and run `hf auth login` first.

| Dataset | qID | Question | Answer |
|---|---|---|:-:|
| HeiCo | 2295666 | There is one Needle in the frame at 01:45:48. When is it retrieved from the surgical site? | `01:48:10` |
| HeiCo | 2362897 | How many Clip(s) are inserted in the abdomen in this video? | `3` |
| LapChole | 1725068 | At what time was a Clip first visible in the video? | `00:16:08` |
| LapChole | 1713930 | What types of foreign objects are seen between 00:23:02 and 00:24:03? | `Clip, Specimen` |

Pass `--ids <dataset>:<id> ...` to use any other SEGMENT test questions.
