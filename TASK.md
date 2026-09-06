You are working directly on the Atlas repository at:

~/atlas_proxy

Your job is to fix and harden the installer/runtime selection system.

Current state

Atlas is an OpenAI/Anthropic-compatible proxy with a CLI and installer.

Relevant files:

* atlas/bin/runtime.py
* atlas/bin/setup_wizard.py
* atlas/bin/atlas
* setup/install.sh

The GitHub remote is:

git@github.com:dommurphy155/Atlas.git

Current branch:

master

The latest remote commit is:

7dba839 fix: make installer runtime privilege-aware

A fresh clone has already been tested.

What currently works

On a normal non-root Ubuntu machine:

Runtime: systemd-user
· wrote ~/.config/systemd/user/atlas-proxy.service
✓ enabled atlas-proxy.service (user)
✓ started atlas-proxy.service (user)
✓ installed CLI
✓ install complete

The user-systemd unit now correctly uses:

WantedBy=default.target

The previous incorrect:

WantedBy=multi-user.target

was fixed for user-systemd.

The Python runtime code has also been made privilege-aware.

Current intended priority is approximately:

1. systemd system
2. systemd-user
3. tmux-compatible runtime
4. nohup
5. manual

Systemd system should work for:

* root directly
* normal users where sudo can authenticate interactively

The old behaviour where installation simply failed because the user wasn’t root must NOT return.

The actual problem

The installer is still too optimistic.

It currently detects a runtime and then largely assumes that runtime will work.

That is NOT good enough.

For example:

systemctl --user exists

does not necessarily mean:

systemd --user actually works

Likewise:

tmux exists

does not necessarily mean Atlas can create and maintain a working tmux session.

The installer must be failure-resistant.

Required architecture

Runtime selection must be:

Detect → Attempt → Start → Verify → Fallback

Do NOT simply:

detect runtime → install runtime → fail installer

Instead:

detect OS/environment
        ↓
build ordered list of usable runtime candidates
        ↓
try candidate #1
        ↓
install
        ↓
start
        ↓
verify Atlas is actually running/healthy
        ↓
SUCCESS → persist runtime and finish
        ↓
FAIL → clean up candidate and try next
        ↓
candidate #2
        ↓
...
        ↓
final manual fallback

The installer itself should only fail for a genuinely fundamental installation failure, such as:

* Python unavailable
* virtualenv cannot be created
* dependencies cannot be installed
* repository is unusable
* Atlas source itself is broken

A supervisor/runtime failure must NOT kill installation.

Required fallback chain

Implement a robust runtime fallback chain based on what actually works on the current machine.

Linux

Prefer:

1. systemd system — if root + systemd is actually usable
2. systemd system via sudo — if non-root + sudo is available and actually usable
3. systemd-user — if genuinely usable
4. tmux-compatible runtime
5. nohup
6. manual foreground execution

macOS

Detect the appropriate native service mechanism if available, then:

1. native service mechanism
2. tmux-compatible runtime
3. nohup
4. manual

Other Unix/POSIX

Use whatever supported service/session mechanism is genuinely available, then:

1. service manager
2. tmux-compatible runtime
3. nohup
4. manual

Windows

Do not break Windows support just because the Linux runtime logic is being changed.

Use the existing/manual mechanism appropriate for Windows.

Do NOT invent a huge Windows service architecture unless the existing project already supports it.

Important: capability detection

Do not consider a runtime usable merely because its binary exists.

For example:

systemd system

Check:

* Linux
* systemctl exists
* PID 1 is actually systemd
* root OR sudo is available
* systemctl can actually communicate with systemd

systemd-user

Check:

* Linux
* systemctl --user exists
* user systemd actually responds
* appropriate user runtime/session environment exists
* installation/start operation actually works

Do not treat:

which systemctl

as sufficient.

tmux family

Existing project logic recognises:

tmux
psmux
tmuxw
lumux
qscn
wmux

Keep that compatibility.

But test the actual ability to create/start a session.

nohup

Verify that the executable exists and that Atlas can actually be launched detached.

manual

Manual is the final safety net.

If nothing else works, installation still completes and tells the user exactly how to start Atlas manually.

Health verification

After starting Atlas through a runtime, verify that Atlas is actually alive.

Use the existing project health mechanism rather than inventing a second one.

The existing proxy health endpoint is:

http://127.0.0.1:8788/health

Allow a short bounded startup window.

Do NOT make the installer sit there for minutes.

Something like a few seconds with short polling is sufficient.

If health succeeds:

runtime = successful

If health fails:

runtime = failed
cleanup
try next runtime

Do not leave broken systemd units, stale tmux sessions, or dead runtime metadata behind when falling back.

Persistence

data/runtime.json is used to remember the selected runtime.

Only persist a runtime after it has actually succeeded.

Do NOT persist:

systemd-user

just because it was detected.

Persist it only after:

1. unit installed
2. service started
3. Atlas health check passed

If everything fails except manual:

{
  "mode": "manual"
}

is acceptable.

Also make get_runtime() resilient.

If runtime.json says:

systemd-user

but systemd-user is no longer available, it should not blindly attempt to use it forever.

It should re-detect/fallback appropriately.

Systemd requirements

System service:

WantedBy=multi-user.target

User service:

WantedBy=default.target

Do NOT globally replace one with the other.

The two unit types have different targets.

System service operations:

systemctl daemon-reload
systemctl enable ...
systemctl start ...

User service operations:

systemctl --user daemon-reload
systemctl --user enable ...
systemctl --user start ...

Do not use sudo for user-systemd.

For normal users, sudo authentication may be interactive.

Do NOT use:

sudo -n

as the only test and incorrectly conclude sudo is unavailable.

setup/install.sh

This is especially important.

./setup/install.sh is independently used by users and must work.

Do not fix only the Python CLI path.

Both:

./setup/install.sh

and:

atlas install

must follow the same runtime philosophy.

Current shell installer behaviour already includes runtime selection and separate:

install_systemd_system()
install_systemd_user()

Keep the existing structure where practical.

Do not rewrite the entire installer.

Make the smallest clean changes necessary.

Speed / development workflow

THIS IS IMPORTANT.

Work in small increments.

Do NOT spend ages performing a giant refactor.

Do NOT redesign unrelated parts of Atlas.

Do NOT touch:

* README unless necessary
* proxy behaviour
* API providers
* model routing
* authentication
* unrelated CLI commands
* frontend
* branding
* documentation

Only work on installer/runtime selection and the minimum supporting code.

After each meaningful change

Run a focused syntax/test check.

For Python:

python3 -m py_compile atlas/bin/runtime.py atlas/bin/setup_wizard.py atlas/bin/atlas

For shell:

bash -n setup/install.sh

Then inspect the diff.

Do NOT launch huge builds.

Do NOT run long-running processes unnecessarily.

Testing strategy

We will manually test this from fresh clones on different environments.

Therefore optimise for quick iteration.

First make the smallest correct change.

Then stop and report:

Changed:
...
Why:
...
Tested:
...
Expected manual test:
...

Do not continue making speculative improvements after the requested change works.

Expected installer UX

A successful runtime selection should look something like:

Detecting runtime
  ✓ systemd-user available
Installing runtime
  ✓ wrote atlas-proxy.service
  ✓ enabled atlas-proxy.service
  ✓ started atlas-proxy.service
  ✓ health check passed
Runtime: systemd-user

If it fails:

Runtime: systemd-user
  ! start failed
Falling back: tmux
  ✓ session created
  ✓ Atlas started
  ✓ health check passed
Runtime: tmux

If tmux also fails:

Falling back: nohup
  ✓ Atlas started
  ✓ health check passed
Runtime: nohup

If literally everything fails:

! No automatic runtime manager available
✓ Atlas installation complete
Manual start:
  ...

The installer must still finish successfully.

Critical invariant

The installer must NEVER get into this situation:

systemd detected
↓
systemd doesn't actually work
↓
installer exits

Instead:

systemd detected
↓
systemd attempt
↓
failure
↓
fallback

Detection is only a candidate-selection mechanism.

Actual successful startup is what makes a runtime valid.

Git

Do NOT push automatically.

After making the change:

git status --short
git diff --check
git diff

Then create a focused commit if the change is correct.

Use a concise commit message such as:

fix: make runtime installation fallback-safe

But DO NOT push unless explicitly instructed.

Final instruction

Start by inspecting the current implementations of:

atlas/bin/runtime.py
atlas/bin/setup_wizard.py
setup/install.sh

especially:

* detect_env
* choose_mode
* install_runtime
* get_runtime
* SystemdRuntime
* SystemdUserRuntime
* TmuxRuntime
* NohupRuntime
* ensure_runtime
* install_systemd_system
* install_systemd_user
* runtime selection in setup/install.sh

Then make the smallest set of changes necessary to implement:

detect → attempt → start → health-check → fallback → persist successful runtime

Do not over-engineer it.

Make one focused change, run the quick checks, and stop so I can manually test it.
