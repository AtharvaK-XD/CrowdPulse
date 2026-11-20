# CrowdPulse report — rules for the Antigravity agent
- You own ONLY: report/, fixtures/, tests/report/, docs/ (except CONTRACT.md),
  .github/, AGENTS.md.
- NEVER edit or import from: crowdpulse/, scripts/, tests/backend/, CLAUDE.md.
- NEVER edit: docs/CONTRACT.md, requirements.txt, config.yaml, .env.example,
  run.py. For README.md, show me the draft and wait for approval.
  If you need a package, STOP and tell me the exact line to add.
- The report reads ONLY the files in data/outputs/<run_id>/ described in
  docs/CONTRACT.md. Never call Apify, Exa or Yahoo from report code.
- Work on git branch `report`. Small commits with clear messages.
- Python 3.11+, Jinja2 + Plotly. The report must be ONE self-contained HTML
  file that opens offline.
