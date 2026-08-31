---
name: desktop-python
description: Build and verify Python desktop apps with a thin launch entry point, testable core logic, and covered UI callbacks.
triggers: tkinter, pyqt, pyside, desktop, gui, window, 桌面, 窗口, 图形界面, 番茄钟
---

For Python desktop applications:

1. Keep the launch entry point thin and place stateful business logic outside the UI layer.
2. Treat UI callbacks as behavior that must be exercised, not as untested glue.
3. Prefer tests that instantiate callback owners with fakes or mocks when a real window cannot be opened.
4. Run import or compile smoke checks for the launch file and UI modules.
5. Report honestly if the GUI could not be manually opened in this environment.
