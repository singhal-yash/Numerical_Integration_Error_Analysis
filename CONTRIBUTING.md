# Contribution Guidelines

## Branch Naming Convention

Use lowercase names with hyphens and include the associated GitHub Issue number.

Examples:

- docs/repo-setup
- feature/trapezoidal-rule
- feature/simpson-rule
- feature/error-calculation
- feature/convergence-order
- feature/experiment-design
- feature/error-plots

## Folder Structure

src/: Numerical algorithms and analysis functions.

tests/: Tests for numerical accuracy and input validation.

examples/: Demonstrations of the implemented methods.

results/: Generated tables and plots.

report/: Final project report.

## Development Workflow

1. Check the assigned GitHub Issue.
2. Update the local 'main' branch.
3. Create a separate branch for the assigned task.
4. Work only on the relevant files.
5. Add tests and run the test suite.
6. Commit changes with meaningful messages.
7. Push the branch and create a Pull Request (PR).
8. Include 'Closes #<issue-number>' in the PR description.
9. Obtain a review from another member before merging.

## Python implementation files
Create files inside 'src/' Initially, function templates are sufficient.
example: trapezoidal.py (For member 1)

## Coding Standards

- Use descriptive 'snake_case' function and variable names.
- Add docstrings to functions.
- Validate inputs and provide clear error messages.
- Keep functions modular and reusable.
- Do not commit virtual environments, cache files, or secrets.
- Keep the 'main' branch functional.

### Authors must not approve their own Pull Requests.
