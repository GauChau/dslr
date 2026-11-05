# Issue tracker: GitHub

Issues and specs live in GitHub Issues for `GauChau/dslr`.
Use the `gh` CLI from this repository.

## Operations

- Verify the target: `gh repo view --json nameWithOwner`.
- Create: `gh issue create --title "..." --body-file <file>`.
- Read: `gh issue view <number> --json number,title,body,labels,comments`.
- List: `gh issue list --state open --json number,title,labels`.
- Label: `gh issue edit <number> --add-label "..."`.
- Remove label: `gh issue edit <number> --remove-label "..."`.
- Comment: `gh issue comment <number> --body-file <file>`.
- Close: post the explanation, then `gh issue close <number>`.

Use native sub-issues and issue dependencies when available.
Otherwise, put `Part of #<parent>` and `Blocked by: #<number>`
at the top of the child issue body.
An issue is unblocked when every blocker is closed.

When a skill says "publish to the issue tracker", create an issue.
When it says "fetch the relevant ticket", read the issue.

Keep the existing `docs/TODO.md`; migration is outside this setup.

## Pull requests as a triage surface

PRs as a request surface: no.
