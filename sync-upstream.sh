#!/bin/bash

# Sync fork with upstream repository
echo "Syncing fork with upstream..."

# Fetch latest changes from upstream
echo "Fetching upstream changes..."
git fetch upstream

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Switch to main if not already there
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "Switching to main branch..."
    git checkout main
fi

# Merge upstream changes
echo "Merging upstream/main into local main..."
git merge upstream/main

# Push updated main to your fork
echo "Pushing updated main to your fork..."
git push origin main

# Switch back to original branch if it wasn't main
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "Switching back to $CURRENT_BRANCH..."
    git checkout "$CURRENT_BRANCH"
fi

echo "Sync complete! Your fork is now up-to-date with upstream."