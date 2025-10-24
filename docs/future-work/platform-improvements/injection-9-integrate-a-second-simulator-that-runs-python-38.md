---
title: "Injection #9: Integrate a second simulator that runs Python >= 3.8"
tags: platform, simulator, environment
skills: python, mujoco
estimated-scope: large
status: in-progress
contributor: jeremyshoemaker
---

If we **(Injection #9 Integrate a second simulator that runs Python >= 3.8)** and we (Injection #16 Integrate a second simulator that runs on CPU) and we (Injection #17 Integrate a second simulator that runs on Windows) and we (Injection #19 Integrate a second simulator that does not require conda), then we achieve the desired effect (141 Monty is decoupled from a specific simulator implementation).

It is worth pointing out that integrating a simulator that runs on Python >= 3.8, on CPU, on Windows, and does not require conda might come down to selecting and integrating a single simulator that meets all of the criteria and not multiple units of work.

![Injection #9](../../figures/future-work/injection_9.png)

## Related work

- [Injection #16: Integrate a second simulator that runs on CPU.](./injection-16-integrate-a-second-simulator-that-runs-on-cpu.md)
- [Injection #17: Integrate a second simulator that runs on Windows.](./injection-17-integrate-a-second-simulator-that-runs-on-windows.md)
- [Injection #19: Integrate a second simulator that does not require conda.](./injection-19-integrate-a-second-simulator-that-does-not-require-conda.md)
