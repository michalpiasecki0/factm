# factm


## Getting started 
```
uv sync
uv shell
```

## Dev
Let's use code checks before commits to make sure code is well-written
Please set up `pre-commit` with:
```
uv run pre-commit install
```
Current checks:
* `black`
* `ruff`
* `mypy`