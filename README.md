# hermes-channel-octo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Octo (WuKongIM-based corporate IM) channel plugin for
[`hermes-agent`](https://github.com/NousResearch/hermes-agent).

Connects a hermes-agent gateway to an Octo bot via the WuKongIM binary
WebSocket protocol (ECDH + AES). Supports bot-to-user DMs, group
messaging, threads, `@`-mentions, and voice / video / file attachments.
Hermes buffers text replies and sends one complete final message.

## Compatibility

| hermes-agent | hermes-channel-octo |
|---|---|
| `>=0.14,<0.21` | `0.1.x` |

## Install

The plugin is verified against `hermes-agent==0.20.0`; the lower bound remains
`0.14` for existing installations. Native interactive `send_clarify` cards are
enabled for Hermes `>=0.20.0`. Hermes `0.14`–`0.19` and unknown/unparseable
versions retain Hermes' plain-text clarify fallback.

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
`aiohttp`, `cryptography`, `python-socks`, `packaging`).

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
             'cryptography>=46.0,<49' 'python-socks>=2.8,<3' \
             'packaging>=24,<27'

```

The `cryptography>=46,<49` range intentionally supports cryptography 46.x and
48.x: 46.x remains compatible with supported older runtimes, while Hermes 0.20
is verified with cryptography 48.0.1. Version 49 is excluded pending a
dedicated compatibility check.

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
| `OCTO_WS_URL` | no | WuKongIM `ws://`/`wss://` override; defaults to the URL returned by bot registration |
| `OCTO_ALLOW_PRIVATE_HOSTS` | no | Set to `true` only for trusted self-hosted API/CDN/WebSocket origins that resolve to private IPs; metadata endpoints remain blocked |

| `OCTO_ON_BEHALF_OF` | no | Trusted grantor user ID for server-authorized persona delivery; text, typing, RichText, and media use this identity, display/interactive Type-17 tools fall back to plain text, and automatic progress cards are disabled |
| `OCTO_ALLOWED_USERS` | no | Comma-separated user IDs allowed to talk to the bot |
| `OCTO_ALLOW_ALL_USERS` | no | Allow any user to trigger the bot (dev only) |
| `OCTO_HOME_CHANNEL` | no | Default group/chat ID for cron / notification delivery |
| `OCTO_CARD_MESSAGE_ENABLED` | no | Legacy `1` opt-in only when the server has no card-profile endpoint; an advertised server manifest remains authoritative |
| `OCTO_EVENT_POLL_INTERVAL_S` | no | Minimum event polling interval in seconds (default `2.0`, minimum `0.5`) |
| `OCTO_EVENT_POLL_WAIT_S` | no | Event long-poll hold in seconds (default `25`, `0` disables, capped at `30`) |
| `OCTO_EVENT_POLL_LIMIT` | no | Events requested per batch (default `50`, clamped to `1..100`) |
| `OCTO_PROGRESS_CARD_RENDERER` | no | Progress-card renderer: `local` (default, Chinese Type-17 execution trace) or `registry` (server `ai.reasoning-process` template when advertised, otherwise local fallback) |
| `OCTO_COMMAND_MENU_MAX_CHARS` | no | Maximum stored JSON characters for the Bot-global command menu; defaults to `1000`, `0` publishes the complete menu, and values `>=2` publish a name-only priority projection that fits the server field |

## Current-conversation tools

When both required credentials are configured, the plugin registers:

- controlled Type-17 display/interactive send and display-card edit tools;
- controlled RichText text/image delivery;
- image, file, voice, and video delivery.

Card capabilities are negotiated automatically; no card-profile diagnostic is
exposed to the model. These tools derive the destination and requester from
Hermes' task-local Octo session, and their schemas accept no channel or identity
overrides. Outbound local media must first pass the installed Hermes runtime's
native media-delivery authorization, then uses inode/no-symlink and 100 MiB
checks before upload. On Hermes 0.14, a missing or rejecting local-media
validator fails closed: the plugin never substitutes its own authorization
decision. HTTP(S) media retains the guarded download flow. Adapter-native Hermes
media delivery also accepts `data:` URLs; the model-facing tools accept HTTP(S),
`file://`, and authorized local paths.

Management audit logs intentionally contain only bounded action, result,
channel-type, and item-count metadata. Stable requester/target identifiers and
model-supplied reason text are omitted to avoid copying cross-channel identity
and content into gateway logs.

For inbound commands, the plugin removes only a leading self-mention immediately
followed by a slash command so Hermes can route that command. Other self-mentions
and every non-command mention remain part of the message text.

After each successful Octo connection, the plugin publishes Octo plugin
commands, the curated Gateway commands `/new`, `/stop`, and `/commands`,
configured quick commands, executable skill bundles, and slash-invocable skills
to the Bot's DM slash-command menu. Other Gateway commands remain available by
manual input and through `/commands`; their names still reserve dispatch
precedence so lower-priority sources cannot publish misleading collisions. The
list is reconciled every minute and after reconnects. Octo's menu is Bot-global,
so command visibility does not imply authorization; Hermes still performs the
normal dispatch, owner, pairing, and disabled-skill checks when a user sends the
selected command. If the deployed Octo Server still uses the legacy
`robot.bot_commands VARCHAR(1000)` schema, keep
`OCTO_COMMAND_MENU_MAX_CHARS=1000` until that column is migrated to `TEXT`/JSON.
Bounded mode publishes empty descriptions and fills the budget in this order:
Octo plugin commands, the three curated Gateway commands, slash skills by usage,
quick commands, other plugin commands, then bundles.

Interactive card actions are accepted only while the originating in-process
card session remains registered and only when message, channel, operator,
action, binding, and Hermes session identity all match. The event cursor is
persisted before acknowledgement. These paths are covered by local automated
tests; production-server card/action/media interoperability still requires the
separately authorized live acceptance checks.

On stable Hermes 0.20.x only, bounded clarifies with choices use the same
trusted current-conversation route and Type-17 session binding. Prerelease,
development, and 0.21+ Hermes versions use the base text fallback. Native
delivery has one 12-second deadline, waits at most 5 seconds for an active
progress card's first send, and rechecks the same pending clarify before every
POST. Single-select prompts render one submit action per choice; multi-select
prompts render `Input.ChoiceSet` plus Submit; both include an **Other** action
that switches the existing clarify to Hermes text capture. Choice clicks call
Hermes' clarify resolution primitive directly and never become a new model turn.
Card/profile/render failures use the base text fallback. Ambiguous POST failures
retry once with the same `client_msg_no` and never send a second prompt. This
version gate is automatic and has no configuration switch.

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
