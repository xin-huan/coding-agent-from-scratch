---
name: diagnose
description: Debug bugs, failures, exceptions, regressions, and performance issues by creating a feedback loop before changing code.
triggers: bug, debug, traceback, exception, error, failure, failed, failing, regression, performance, slow, 报错, 错误, 异常, 失败, 崩溃, 卡住, 性能, 很慢, 复现, 修复, 测试不通过
---

Work diagnostically:

1. Establish a fast feedback loop that reproduces or detects the reported symptom before making broad changes.
2. Confirm the observed failure matches the user's report.
3. State and test a focused hypothesis using the smallest useful probe.
4. Apply the minimal fix and add or update a regression test at the closest meaningful seam.
5. Re-run the original failing path and the relevant test suite before reporting completion.

If the symptom cannot be reproduced in this environment, say exactly what was checked and validate the nearest automated seam instead of guessing.
