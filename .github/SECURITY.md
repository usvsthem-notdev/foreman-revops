# Security Policy

## Scope

Foreman RevOps Tracker is a **local-first** tool. All data is stored in a SQLite
file on your machine. No telemetry, no external API calls, no user accounts.

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Email: founders@[your-domain].com  
Subject: `[SECURITY] foreman-revops — <brief description>`

We will acknowledge within 48 hours and aim to ship a patch within 14 days.

## Security design notes

- **SQL injection**: all user-supplied values are bound via `?` parameters.
  WHERE clause skeletons are assembled from hard-coded string literals, never
  from user input.
- **File uploads**: the declared file size (`UploadedFile.size`) is checked
  before `read()` is called. Content is then decoded as strict UTF-8 and
  rejected (raises `ValueError`) if it fails — `errors="replace"` is not used.
  No files are written to disk from uploads.
- **Path traversal**: `FOREMAN_DB_PATH` is validated with `Path.is_relative_to()`
  (boundary-aware, not string prefix matching) against the user's home directory,
  the system temp dir, and `/tmp`. Symlinks are resolved before comparison.
- **XSS**: user-sourced strings (e.g. model names from uploaded CSVs) are escaped
  with `html.escape()` before being embedded in any `unsafe_allow_html` block.
- **No outbound network calls**: the app does not call any external API. The
  billing CSVs are parsed entirely in-process.
- **Docker**: the container runs as a non-root user (`foreman`). The security
  flags `no-new-privileges` and `read_only: true` are enforced in
  `docker-compose.yml`. Users running the image directly with `docker run`
  should add `--security-opt no-new-privileges --read-only --tmpfs /tmp`.
