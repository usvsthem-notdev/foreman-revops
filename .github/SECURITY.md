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

- **SQL injection**: all database queries use parameterized statements. No
  string interpolation is used in SQL.
- **File uploads**: size is validated (50 MB limit) before reading. Content is
  decoded as UTF-8 and rejected if it fails. No files are written to disk from
  uploads.
- **Path traversal**: `FOREMAN_DB_PATH` is validated to be within the user's home
  directory or `/tmp`.
- **No outbound network calls**: the app does not call any external API. The
  billing CSVs are parsed entirely in-process.
- **Docker**: the container runs as a non-root user (`foreman`) with
  `no-new-privileges` and a read-only root filesystem.
