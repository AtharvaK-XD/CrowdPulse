# CrowdPulse backend — rules for Claude Code
- You own ONLY: crowdpulse/, scripts/, tests/backend/, CLAUDE.md.
- NEVER edit: report/, fixtures/, docs/, .github/, tests/report/, AGENTS.md,
  README.md, requirements.txt, config.yaml, .env.example, run.py.
  If you need a package or a config key, STOP and tell me the exact line to add.
- Output formats are defined in docs/CONTRACT.md. Follow them exactly.
- NEVER run git commit, git push, or any other command that writes to git history
  or GitHub. Git is handled by Antigravity. Leave changes uncommitted in the
  working tree and list the changed files at the end of each phase.
- Python 3.11+, type hints, pandas. Read secrets from .env via python-dotenv.
- Every module gets a pytest test in tests/backend/ that runs OFFLINE
  (use saved sample JSON in tests/backend/samples/; never call live APIs in tests).
- No look-ahead: features for day t may only use data available before day t's close.
- Never import anything from report/.
