TESTING DIRECTORY (`tests/`)

Purpose:
This directory contains automated tests to verify the correctness and reliability of the implemented numerical methods.

Planned files:
1. test_trapezoidal.py — Tests for Member 1's implementation.
2. test_simpson.py — Tests for Member 2's implementation.
3. test_error_analysis.py — Tests for Member 3's error calculations.

Testing requirements:
- Check numerical results against known exact integrals.
- Test different intervals and subdivision counts.
- Verify input validation and invalid-input handling.
- Check that Simpson's 1/3 Rule rejects odd or non-positive subdivision counts.
- Test error calculations and convergence order estimates.
