#!/usr/bin/env node
// SessionStart hook for Graft (https://github.com/trailhq/Graft).
//
// Cloud sessions start from a fresh clone, so graft/ (git-ignored) is missing.
// Before handing off to graft's own session-start, this runs, in order:
//   1. npm install -g @nanonets/graft      (only if the environment didn't install it)
//   2. graft telemetry disable
//   3. graft init --yes --no-global --no-agents --no-statusline
//        (--no-global keeps hooks and the MCP server out of ~/.claude and ~/.claude.json,
//         trailhq/Graft#491, #497)
//   4. graft build                          (structural graph, no LLM, no API key)
// Locally it only runs graft's session-start, which no-ops if graft isn't installed.
//
// `graft init` rewrites tracked wiring files (.claude/settings.json re-gains graft's own
// SessionStart entry, .mcp.json loses the graft-mcp.cjs wrapper, the helpers and the graft
// skill are regenerated), so this hook puts back every tracked file init changed. The
// session therefore starts with a clean tree, and this hook stays the only SessionStart entry.
const path = require('path');
const { execSync, spawnSync } = require('child_process');

const dir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
const quiet = (cmd) => execSync(cmd, { cwd: dir, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
// Tracked files with uncommitted changes (porcelain: "XY path"; untracked "??" excluded).
const modified = () =>
  new Set(quiet('git status --porcelain --untracked-files=no').split('\n')
    .filter(Boolean).map((l) => l.slice(3)));

if (process.env.CLAUDE_CODE_REMOTE === 'true') {
  try {
    try {
      quiet('graft --version');
    } catch {
      quiet('npm install -g @nanonets/graft');
    }
    quiet('graft telemetry disable');
    const before = modified();
    try {
      quiet('graft init --yes --no-global --no-agents --no-statusline');
    } finally {
      const touched = [...modified()].filter((f) => !before.has(f));
      if (touched.length) {
        execSync(`git checkout -- ${touched.map((f) => JSON.stringify(f)).join(' ')}`,
          { cwd: dir, stdio: 'ignore' });
      }
    }
    quiet('graft build');
  } catch {
    // Graft is an optimization; never block the session over it.
  }
}

const r = spawnSync(process.execPath, [path.join(dir, '.claude', 'helpers', 'graft-hooks.cjs'), 'session-start'], {
  stdio: 'inherit',
});
process.exit(r.status ?? 0);
