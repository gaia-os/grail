# Formatting, Linting, and Style

We will use [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting.

### Adding rules

I found an interesting command in the docs for ruff. If you want to add a new rule but allow existing code to be an exception (so that you don't have to deal with potentially many flags), you can run

```bash
ruff check --select <code> --add-noqa
```

This will add the `noqa` indicator on all the flagged lines to exempt them in the future.  
See [https://docs.astral.sh/ruff/tutorial/#adding-rules](here) for more.