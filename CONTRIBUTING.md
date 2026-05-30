# Contributing to hermes-channel-octo

Thanks for taking the time to contribute! This project is an Octo (WuKongIM)
platform channel plugin for [`hermes-agent`](https://github.com/NousResearch/hermes-agent).

## Development setup

1. Fork the repo at https://github.com/Mininglamp-OSS/hermes-channel-octo
   and clone your fork:

   ```bash
   git clone https://github.com/<your-user>/hermes-channel-octo.git
   cd hermes-channel-octo
   ```

2. Create a virtual environment and install the package with dev extras:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

   This pulls in `pytest`, `pytest-asyncio`, `pytest-timeout`, and `ruff`
   alongside the runtime dependencies.

## Running checks

Before opening a pull request, make sure both checks pass:

```bash
ruff check .
pytest
```

CI runs the same two commands across Python 3.11 and 3.12.

## Code style

- Match the existing style. The repo enforces only one ruff rule
  (`PLW1514` — unspecified text encoding), but we follow standard PEP 8
  conventions and prefer self-documenting names over comments.
- Keep changes surgical. Don't refactor unrelated code in the same PR.
- Add or update tests for any behavior change.

## Commit messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) style:

```
feat: add per-thread member cache
fix: drop stale uid index entries on member removal
docs: clarify Octo channel_type=5 (thread) semantics
chore: bump hermes-agent ceiling to <0.16
```

Common prefixes: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`.

## Pull request flow

1. Create a branch off `main`:
   `git checkout -b feat/short-description`
2. Make your change with focused commits.
3. Run `ruff check .` and `pytest` locally.
4. Push to your fork and open a PR against `Mininglamp-OSS/hermes-channel-octo:main`.
5. Describe what changed, why, and how you verified it. Link any related issues.

A maintainer will review and merge once CI is green and feedback is addressed.

## Reporting issues

Use the [issue tracker](https://github.com/Mininglamp-OSS/hermes-channel-octo/issues).
For bugs, please include:

- hermes-agent and hermes-channel-octo versions
- Python version and OS
- A minimal reproduction or relevant gateway log excerpt (scrub bot tokens)
