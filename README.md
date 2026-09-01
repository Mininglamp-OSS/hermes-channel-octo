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
disabled on every supported Hermes version; clarifies use Hermes' plain-text
fallback so users are never required to resolve a card action.

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
| `OCTO_BOT_TOKEN` | yes | Octo bot authentication token; several tokens may be given, separated by `;` (see [Multiple bot identities](#multiple-bot-identities)) |
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

## Multiple bot identities

Octo issues a separate bot token per Space. One `octo` platform can hold
several of them:

```dotenv
OCTO_BOT_TOKEN="bf_space_one;bf_space_two"
```

Before enabling a semicolon-separated token list, start this plugin successfully
once with the profile's existing token alone. That one-token start durably
records the bot's stable `robot_id`; startup refuses to guess legacy ownership
without it. Then add the other tokens, keeping that existing token first for the
one-time migration. A rotated replacement is also valid if it registers as the
same `robot_id`. After migration reaches `phase=migrated`, token order no longer
matters.

Each token registers on its own, keeps its own WebSocket, heartbeat, event
cursor, caches and card bindings, and reconnects independently. Empty and
repeated entries are configuration errors and are reported without echoing
any token.

Configure at most one token from any one Octo Space in the same profile. A
Space-prefixed DM id is scoped to its Space, not to an individual bot, so two
identities from the same Space can derive the same Hermes SessionKey. Same-Space
multi-token profiles are unsupported; use separate Hermes profiles for them.

All identities share one Hermes profile: the same model, persona, memory,
`session_list`, tools and filesystem. Nothing identity-specific reaches the
model — tool schemas expose no token, Space or bot field. Existing legacy
primary and group SessionKeys stay unchanged. If Octo supplies the same bare DM
uid in several Spaces, non-primary identities receive an opaque internal DM
scope derived from the stable `robot_id`; the route separately persists the
bare wire uid, and no token or raw `robot_id` enters the SessionKey.

Routing is durable, not guessed. The first message of a conversation binds
that conversation to the identity that received it, keyed by the bot's stable
`robot_id`; every later reply, progress update, card, tool message and media
send goes back through the same identity. Consequences:

- rotating a token keeps every existing private chat, group and session, because
  the replacement token registers as the same `robot_id`;
- after the first migration, reordering `OCTO_BOT_TOKEN` changes nothing;
- an identity that is offline or no longer configured makes its conversations
  fail with an explicit error instead of silently answering as another bot;
- proactive cron/notification delivery requires an established route; a newly
  added group or peer must send one trusted inbound message before the bot can
  speak first, and a migrated DM whose wire peer is not yet known follows the
  same fail-closed rule;
- a profile that already ran a single token migrates its existing sessions to
  that bot once, on the first multi-token start.

The platform is online while at least one identity is connected; `/octo_doctor`
reports per-identity status. Routing state lives in
`$HERMES_HOME/workspace/octo/identity/`, and per-identity card bindings and
event cursors in `$HERMES_HOME/workspace/octo/<robot_id>/`. Neither stores a
token or a token hash.

The durable registry holds at most 4096 target routes. `/octo_doctor` reports
the configured capacity, remaining slots, and exhaustion state. At capacity,
new conversations are refused without evicting established ownership. Within
the supported one-token-per-Space configuration, the bot owner may reclaim one
confirmed-stale, unconflicted DM route with
`/octo-route-forget dm <chat_id>`. Releasing a group route removes its linked
SessionRoutes but does not delete the Hermes session or transcript. A later
inbound handled by another Octo identity can therefore claim that same
unscoped group SessionKey and inherit its existing transcript. Group release
requires explicit acknowledgement:
`/octo-route-forget group <chat_id> --confirm-transcript-inheritance`.
Successful output is returned only after the route snapshot is durable.

The identity state directory must be writable even for a one-token profile:
startup records a sentinel before accepting inbound traffic so a later
multi-token migration cannot guess ownership.

If startup reports unusable/pending identity state, or a route conflict remains
after fixing the token configuration, stop the gateway before recovery. First
restore the token list that created the pending migration, with the original
token first. If that is impossible, back up and then remove both
`$HERMES_HOME/workspace/octo/identity/` and the affected
`$HERMES_HOME/workspace/octo/<robot_id>/card-sessions.json` shards. Restart once
with the original token alone before adding other tokens again. This destructive
reset discards pending card actions and durable route ownership; conversations
bind again only from new trusted inbound messages.

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

Native clarify cards are disabled on every supported Hermes version. Clarify
prompts use Hermes' base text fallback, and typed replies are resolved by Hermes'
existing clarify intercept. In groups and topics with `require_mention` enabled,
the prompt tells the user to mention the bot so Octo dispatches the reply through
the normal mention gate. Other interactive cards remain governed by the session
binding and event rules above.

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
