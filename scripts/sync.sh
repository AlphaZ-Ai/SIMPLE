#!/usr/bin/env bash
# Copyright (c) 2025-2026 The SIMPLE Authors
# SPDX-License-Identifier: MIT

# Sync script: Synchronize files from SRC to DST, excluding venv, data, and third_party

set -e

# Configuration
SRC="${1:-.}"      # Source directory (default: current directory)
DST="${2:-.}"      # Destination directory (required if SRC is provided)

# Help message
show_help() {
    cat << EOF
Usage: $0 [SRC] [DST] [OPTIONS]

Synchronize files from SRC to DST folder, excluding:
  - .venv*/
  - data/
  - third_party/
  - typings/
  - */.doctrees/
  - cache/*
  - .uv-cache/
  - .vscode/
  - output/*
  - .mypy_cache/
  - */.cache/*
  - docs/build/*
  - .claude/
  - .runtime-state/
  - scripts/sync.sh

Arguments:
  SRC           Source directory (default: current directory)
  DST           Destination directory (required if SRC is provided)

Options:
  --dry-run        Show what would be synced without making changes
  --delete         Delete files in DST that don't exist in SRC
  --no-submodules  Skip syncing submodule commits (DST must be a git clone)
  --help           Show this help message

Examples:
  # Sync current directory to remote
  $0 . /mnt/backup

  # Dry run to preview changes
  $0 . /mnt/backup --dry-run

  # Sync with deletion
  $0 . /mnt/backup --delete

EOF
}

# Parse arguments
DRY_RUN=""
DELETE=""
SYNC_SUBMODULES="1"

for arg in "$@"; do
    case "$arg" in
        --help)
            show_help
            exit 0
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            ;;
        --delete)
            DELETE="--delete"
            ;;
        --no-submodules)
            SYNC_SUBMODULES=""
            ;;
    esac
done

# Validate arguments
if [ "$SRC" = "--dry-run" ] || [ "$SRC" = "--delete" ] || [ "$SRC" = "--help" ]; then
    SRC="."
fi

if [ -z "$DST" ] || [ "$DST" = "--dry-run" ] || [ "$DST" = "--delete" ]; then
    if [ "$SRC" = "." ]; then
        show_help
        exit 1
    fi
fi

# Resolve absolute paths
SRC="$(cd "$SRC" 2>/dev/null && pwd)" || { echo "❌ Source directory not found: $SRC"; exit 1; }
DST="$(cd "$(dirname "$DST")" 2>/dev/null && pwd)/$(basename "$DST")" || { echo "❌ Destination parent not found"; exit 1; }

# Create destination if it doesn't exist
mkdir -p "$DST"

echo "📁 Sync Configuration"
echo "===================="
echo "Source:      $SRC"
echo "Destination: $DST"
echo "Excluding:   .venv*/, data/, third_party/, typings/, */.doctrees/, cache/*, .uv-cache/, .vscode/, output/*, .mypy_cache/, */.cache/*, docs/build/*, .claude/, .runtime-state/, scripts/sync.sh"
[ -n "$DRY_RUN" ] && echo "Mode:        🔍 DRY RUN (no changes)"
[ -n "$DELETE" ] && echo "Delete:      ✓ Enabled"
echo ""

# Rsync options
RSYNC_OPTS=(
    -av                           # Archive mode, verbose
    --progress                    # Show progress
    --exclude='.venv*'            # Exclude venv and variants
    --exclude=data                # Exclude data folder
    --exclude=third_party         # Exclude third_party folder
    --exclude=typings             # Exclude typings folder
    --exclude='*/.doctrees'       # Exclude sphinx doctrees
    --exclude='cache/*'           # Exclude cache folder contents
    --exclude=.uv-cache           # Exclude uv package cache
    --exclude=.vscode             # Exclude vscode settings
    --exclude='output/*'          # Exclude output folder contents
    --exclude=.mypy_cache         # Exclude mypy cache
    --exclude='*/.cache/*'        # Exclude nested .cache directories
    --exclude='docs/build/*'      # Exclude built documentation
    --exclude=.claude             # Exclude claude configuration
    --exclude=.runtime-state      # Exclude runtime state
    --exclude='scripts/sync.sh'    # Exclude sync script itself
    --exclude=.git                # Exclude git
    --exclude=.gitignore          # Exclude gitignore
    --exclude=.env                # Exclude env files
    --exclude='*.pyc'             # Exclude Python cache
    --exclude=__pycache__         # Exclude Python cache directory
    --exclude=.pytest_cache       # Exclude pytest cache
    --exclude='*.egg-info'        # Exclude egg info
)

# Add optional flags
[ -n "$DRY_RUN" ] && RSYNC_OPTS+=(--dry-run)
[ -n "$DELETE" ] && RSYNC_OPTS+=(--delete)

# Sync submodule commits: force each DST submodule to the exact commit
# currently checked out in SRC. rsync excludes .git, so submodule pointers
# are git metadata that must be replicated explicitly.
sync_submodules() {
    if [ ! -d "$SRC/.git" ] && [ ! -f "$SRC/.git" ]; then
        echo "⚠️  Source is not a git repo; skipping submodule sync."
        return 0
    fi
    if [ ! -d "$DST/.git" ] && [ ! -f "$DST/.git" ]; then
        echo "⚠️  Destination is not a git clone; skipping submodule sync."
        echo "    (run 'git clone' there first, or pass --no-submodules)"
        return 0
    fi

    echo ""
    echo "🔗 Syncing submodule commits..."

    # Each line: "<submodule_path> <checked-out SHA in source>"
    git -C "$SRC" submodule foreach --quiet 'echo "$sm_path $(git rev-parse HEAD)"' \
    | while read -r sm_path sha; do
        [ -z "$sm_path" ] && continue
        if [ -n "$DRY_RUN" ]; then
            echo "  [dry-run] $sm_path -> $sha"
            continue
        fi

        echo "  $sm_path -> $sha"
        # Ensure the submodule exists/initialized on the destination.
        git -C "$DST" submodule update --init -- "$sm_path" >/dev/null 2>&1 || true

        if [ ! -e "$DST/$sm_path/.git" ]; then
            echo "    ⚠️  not initialized on destination; skipping"
            continue
        fi

        # Make sure the target commit is available, then check it out.
        if ! git -C "$DST/$sm_path" cat-file -e "${sha}^{commit}" 2>/dev/null; then
            git -C "$DST/$sm_path" fetch --all --quiet 2>/dev/null || true
        fi
        if git -C "$DST/$sm_path" checkout --quiet "$sha" 2>/dev/null; then
            # Record the gitlink in the superproject so DST status is clean.
            git -C "$DST" add "$sm_path" 2>/dev/null || true
        else
            echo "    ❌ commit $sha not found in $sm_path (fetch may have failed)"
        fi
    done
    echo "  ✓ Submodule commits synced"
}

# Run rsync
echo "🔄 Syncing files..."
echo ""

if rsync "${RSYNC_OPTS[@]}" "$SRC/" "$DST/"; then
    [ -n "$SYNC_SUBMODULES" ] && sync_submodules
    echo ""
    echo "✅ Sync completed successfully!"
    if [ -n "$DRY_RUN" ]; then
        echo "💡 Run without --dry-run to apply changes"
    fi
else
    echo ""
    echo "❌ Sync failed!"
    exit 1
fi
