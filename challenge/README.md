# Challenge container

`inference.py` is the script of our submitted image (pre-evaluation score 0.6346), with comments
rewritten. It reads `/input/request.json`, `/input/FO_definitions.json` and
`/input/overlayed/<qID>.mp4`, and writes `/output/answer.json`.

```bash
python challenge/fetch_resources.py        # base model + adapter
docker build --platform linux/amd64 -t surgzoom-segment challenge/
docker run --rm --gpus all -v $PWD/examples/demo:/input:ro -v $PWD/out:/output surgzoom-segment
```
