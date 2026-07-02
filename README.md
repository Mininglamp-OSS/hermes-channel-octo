# hermes-channel-octo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Octo (WuKongIM-based corporate IM) channel plugin for
[`hermes-agent`](https://github.com/NousResearch/hermes-agent).

Connects a hermes-agent gateway to an Octo bot via the WuKongIM binary
WebSocket protocol (ECDH + AES). Supports bot-to-user DMs, group
messaging, threads, `@`-mentions, voice / video / file attachments, and
streaming responses.

## Companion X/Twitter Notifications

For corporate monitoring workflows, pair Octo with
[Hermes Tweet](https://github.com/Xquik-dev/hermes-tweet). Hermes Tweet can
collect the X/Twitter context inside Hermes, then Octo can route the resulting
summary or follow-up request to a user, group, or home channel.

## Compatibility

| hermes-agent | hermes-channel-octo |
|---|---|
| `>=0.14,<0.16` | `0.1.x` |

## Install

The plugin supports two install paths. Both have been end-to-end
verified against `hermes-agent==0.15.2`.

All commands below assume `HERMES_HOME` points at the hermes install you
want to wire the plugin into, and that you invoke the matching `hermes`
binary from that install's venv:

```bash
export HERMES_HOME=~/.hermes              # adjust to your install
HERMES=$HERMES_HOME/.venv/bin/hermes
PIP=$HERMES_HOME/.venv/bin/pip
```

### Recommended: pip (entry-point discovery)

```bash
# From GitHub:
$PIP install 'git+https://github.com/Mininglamp-OSS/hermes-channel-octo.git'

# From PyPI (once published):
# $PIP install hermes-channel-octo
```

Pip resolves all runtime dependencies automatically (`websockets`,
`aiohttp`, `cryptography`, `python-socks`).

The plugin is registered via Python entry-points and **loads on the
next gateway start** — no `hermes plugins enable` needed. Note that
entry-point plugins do **not** show up in `hermes plugins list` (which
only lists directory-scanned plugins); confirm load via gateway logs
(see *Verify* below).

### Alternative: `hermes plugins install` (bundled clone)

```bash
$HERMES plugins install Mininglamp-OSS/hermes-channel-octo
$HERMES plugins enable octo

# bundled-plugin protocol does NOT install pyproject deps — install manually:
$PIP install 'websockets>=15.0,<16' 'aiohttp>=3.13,<4' \
             'cryptography>=46.0,<47' 'python-socks>=2.8,<3'
```

`hermes plugins install` clones into `$HERMES_HOME/plugins/octo/` (the
directory name comes from `plugin.yaml`'s `name:` field, not the repo
name). Bundled plugins are opt-in, so the explicit `enable` step is
required.

Prefer the pip path unless you need the in-tree clone for local hacking.

## Configuration

Set the following in `$HERMES_HOME/.env` (or via `hermes config`):

| Variable | Required | Purpose |
|---|---|---|
| `OCTO_API_URL` | yes | Octo bot API base URL (e.g. `https://api.botgate.cn`) |
| `OCTO_BOT_TOKEN` | yes | Octo bot authentication token |
| `OCTO_CDN_URL` | no | CDN prefix for media acceleration |
| `OCTO_ALLOWED_USERS` | no | Comma-separated user IDs allowed to talk to the bot |
| `OCTO_ALLOW_ALL_USERS` | no | Allow any user to trigger the bot (dev only) |
| `OCTO_HOME_CHANNEL` | no | Default group/chat ID for cron / notification delivery |

## Start / Verify

```bash
$HERMES gateway restart
tail -f $HERMES_HOME/logs/gateway.log
```

Successful load looks like:

```
INFO gateway.run: Connecting to octo...
INFO hermes_octo_plugin.adapter: [Octo] Bot registered: robot_id=...
INFO hermes_octo_plugin.adapter: [Octo] Connected (server_version=4)
INFO gateway.run: ✓ octo connected
INFO gateway.run: Gateway running with 1 platform(s)
```

If you see `No messaging platforms enabled`, the plugin did not load.
Common causes:

- pip path: gateway was already running when the package was installed —
  always `gateway restart` after pip install.
- bundled path: forgot `hermes plugins enable octo`, or forgot to install
  the runtime deps listed above.

## License

MIT — see [`LICENSE`](./LICENSE). Portions adapted from
[`NousResearch/hermes-agent`](https://github.com/NousResearch/hermes-agent)
(MIT, Copyright (c) 2025 Nous Research).
