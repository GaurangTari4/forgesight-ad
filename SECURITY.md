# Security policy

## Supported scope

ForgeSight is a local research prototype. Its Docker Compose deployment binds
the UI to `127.0.0.1` and uses `share=False`; it is not designed as an
internet-facing service.

## Reporting a vulnerability

Do not include credentials, proprietary images, artifacts, or exploit details
in a public issue. Until a project security contact is published, report a
minimal reproducible description privately to the repository owner through the
GitHub contact mechanism. Include affected revision, environment, impact, and
safe reproduction steps.

## Operational expectations

- Treat checkpoints and mounted artifacts as trusted local inputs only.
- Keep artifact mounts read-only and verify provenance/checksums before use.
- Do not expose the Gradio port beyond loopback without a separate security
  design, authentication, and network review.
- Do not store API keys, datasets, model weights, or experiment outputs in Git.
- Run dependency and container-vulnerability scans in the target publication or
  deployment environment; an SBOM and vulnerability review are not yet part of
  the verified release evidence.
