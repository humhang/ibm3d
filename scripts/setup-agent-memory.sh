#!/usr/bin/env bash
#
# Link the in-repo .agent-memory/ directory to where Claude Code looks for
# per-project memory on this machine.  Idempotent — safe to re-run.
#
# Claude Code keys memory by the project's absolute path with '/' replaced
# by '-'.  So a repo at /home/alice/work/ibm3d resolves to
#   ~/.claude/projects/-home-alice-work-ibm3d/memory
# which we make a symlink → <repo>/.agent-memory.

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
encoded="$(printf '%s' "$repo" | tr / -)"
link_parent="$HOME/.claude/projects/$encoded"
link="$link_parent/memory"
target="$repo/.agent-memory"

if [[ ! -d "$target" ]]; then
    echo "Missing $target — is this a fresh clone?  Nothing to link." >&2
    exit 1
fi

mkdir -p "$link_parent"

if [[ -L "$link" ]]; then
    current="$(readlink "$link")"
    if [[ "$current" == "$target" ]]; then
        echo "Already linked: $link -> $current"
        exit 0
    fi
    echo "Replacing existing symlink ($current → $target)"
    rm "$link"
elif [[ -d "$link" ]]; then
    backup="$link.backup.$(date +%Y%m%d-%H%M%S)"
    echo "Existing directory at $link — backing up to $backup"
    mv "$link" "$backup"
elif [[ -e "$link" ]]; then
    echo "Refusing to overwrite non-directory at $link" >&2
    exit 1
fi

ln -s "$target" "$link"
echo "Linked $link -> $target"
