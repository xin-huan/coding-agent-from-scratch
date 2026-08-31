---
name: verification
description: Make delivery harder to fake by requiring explicit acceptance checks, automated tests, and real launch or smoke commands.
triggers: 创建, 实现, 添加, 修改, 修复, 完成, 交付, 测试, 验证, build, create, implement, add, fix, test, verify
---

Before reporting completion:

1. Compare the finished behavior against each explicit user requirement.
2. Add or update automated tests for the changed behavior when there is a meaningful seam.
3. Run the complete relevant test suite after the latest file change.
4. Run a launch, import, compile, or API smoke check when the deliverable has an entry point.
5. Include the exact command results and launch instructions in the final report.
