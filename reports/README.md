# Validated-platform baselines

Committed `hanajit doctor` reports from real hardware, one per platform.
These are historical evidence, not something your run produced — when YOU
run `python -m hanajit.doctor`, it writes a fresh report for *your*
platform into your current directory.

| report | machine | result |
|---|---|---|
| hanajit_report_linux_x86_64.md | Linux, Cascade Lake, py3.12 | 18 pass / 0 fail |
| hanajit_report_windows_amd64.md | Windows 11, 16-core, py3.14 | 18 pass / 0 fail |
| hanajit_report_darwin_arm64.md | macOS, Apple M4, py3.14 | 19 pass / 0 fail (incl. real `xcrun metal` -> AIR) |
