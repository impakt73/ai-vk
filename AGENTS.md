# Commit Messages

When asked to create a git commit, inspect the complete set of staged and
unstaged changes, then review the surrounding code and repository context.
Write the commit message from that analysis rather than using a generic or
mechanically generated summary.

The commit message must:

- Use a concise title of no more than 50 characters.
- Put a blank line between the title and the body.
- Explain what changed and why the change was made.
- Include relevant implementation, behavior, or compatibility details when
  they help a reviewer understand the commit.
- Wrap every body line at no more than 72 characters.

The title should describe the primary outcome of the commit in imperative
language when practical. Keep the body focused on the intent and meaningful
effects of the complete change, not a file-by-file inventory. Before creating
the commit, verify the title length, blank-line separation, and body wrapping.
