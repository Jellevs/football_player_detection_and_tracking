# Claude Coding Rules

---
paths: ["**/*.py", "**/*.js", "**/*.md", "**/*.ipynb"]
---


## Enforcement
These rules are defaults, not suggestions. Follow them on every task in this repo unless the user's current prompt explicitly overrides a specific rule. If a rule and a user request conflict, ask before breaking the rule and do not silently override it.

## Philosophy
Write code that a colleague can understand in one sitting. Prioritize clarity and simplicity over cleverness. Code should look like it was written by a careful human, not generated. Ask clarifying questions before writing to ensure alignment.

## Before Writing Code
1. Ask clarifying questions about requirements, constraints, and edge cases.
2. Confirm the scope and desired approach with the user.
3. Never assume intent - verify assumptions explicitly.
4. State what you will do before writing a single line.
5. Think of making it very configurable so that adding, updating things is easy.
6. Before editing any file, read the entire file - not a snippet or a search match. Read every other file in the repo that the change touches or depends on (imports, callers, shared config). Do this so you do not duplicate something that already exists elsewhere in the codebase.

## Language and Stack Constraints
1. Only use the language(s) already present in the repo. If the repo is Python, write Python - no mixing in other languages or runtimes.
2. If the repo is HTML, CSS, and vanilla JS, keep it that way. Do not introduce TypeScript, build tools, bundlers, or JS frameworks/libraries unless the user explicitly asks for them.
3. Do not add a new dependency, package, or library to solve something that can be done with what is already in the repo or the language's standard library.
4. If a task seems to require a new language, framework, or dependency, stop and ask the user first instead of adding it.

## Naming
1. Use full, descriptive names. Never abbreviate: `config` not `cfg`, `temporary` not `tmp`, `result` not `res`, `file` not `fp`.
2. Names should reveal intent. A reader should understand what a variable holds without context. No ambiguous names - if a name could mean two different things, rename it.
3. Use active verbs for functions: `calculate_total()` not `process()`, `validate_input()` not `check()`.

## Code Structure
1. Keep functions small and focused. If a function needs many parameters, break it into smaller pieces.
2. Avoid nested logic deeper than two levels. Refactor if you exceed this.
3. Use early returns to flatten control flow. Avoid deep if-else chains.
4. One concept per function. If you describe it with "and" or "then", split it.

## Comments
1. Only comment the "why", never the "what". Code that needs "what" comments is unclear code.
2. Explain non-obvious decisions: performance tradeoffs, workarounds for bugs, or surprising behavior.
3. No separator lines, banners, or decorative comments. Ever.
4. No commented-out code. Delete it.

## Print Statements and Output
1. Use print only for operational information: status, progress, or warnings.
2. Never use print for visual spacing, indentation, or decoration.
3. Never use print to debug. Use a debugger or structured logging.

## Simplicity
1. Choose the simplest solution first. Add complexity only if necessary.
2. Avoid unnecessary helper functions, wrapper functions, or abstraction layers.
3. Avoid indirection. If code is clear, inline it.
4. No patterns, frameworks, or idioms unless they solve a real problem in the code.

## Readability
1. Code should be easy to follow from top to bottom.
2. Variable scope should be as small as possible.
3. Avoid side effects. Functions should do one thing.
4. Make the happy path obvious. Put error handling or edge cases after.

## Efficiency (Tokens and Time)
1. Read a file in full before editing it. Once read in the current task, do not re-read it again unless it changed or correctness requires it.
2. Do not repeat work. If something is done, reference it.
3. Keep edits focused. Do not refactor code unrelated to the task.
4. Ask before starting, not after. Clarify once, code once.

## Appearance
1. Code should look like a human wrote it. No weird ASCII art or formatting tricks.
2. Indentation and whitespace should follow standard conventions for the language.
3. Use language idioms, but only the readable ones.