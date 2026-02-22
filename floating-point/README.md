# 🧮 Floating-Point Arithmetic Explorer

An interactive, visual notebook that demystifies how computers handle real numbers, heavily inspired by David Goldberg's classic paper, *"What Every Computer Scientist Should Know About Floating-Point Arithmetic"*.

This tool allows you to instantly explore the IEEE 754 binary representations of decimals across modern computing formats like **FP64**, **FP32**, **BF16**, and NVIDIA's **TF32**. It also features interactive demonstrations of mathematical quirks like absolute vs. relative rounding error and Catastrophic Cancellation.

Built with [Marimo](https://marimo.io/) and Python.

## 🚀 Quick Start (Recommended)

You don't need to manually configure anything. The provided start scripts will automatically create an isolated virtual environment, install `marimo` and `numpy`, and launch the web app locally.

**Prerequisites:** You must have Python 3.10+ installed on your system.

### For Mac / Linux
Open your terminal, navigate to this directory, and run:
```bash
chmod +x start.sh
./start.sh