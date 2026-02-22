# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.10",
#     "pyzmq>=27.1.0",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


@app.cell
def imports():
    import marimo as mo
    import struct
    import numpy as np
    from decimal import Decimal
    from dataclasses import dataclass
    from typing import Tuple, List, Optional

    return Decimal, List, Optional, Tuple, dataclass, mo, np, struct


@app.cell
def title_and_toc(mo):
    mo.md("""
    # 🧮 What Every Computer Scientist Should Know About Floating-Point Arithmetic
    *(An Interactive Visualizer)*

    ### 📖 About this Notebook
    
    Based on the classic paper by David Goldberg, this interactive notebook demystifies how computers handle real numbers and the mathematical quirks that arise from finite memory.

    This interactive notebook is designed to demystify floating-point numbers, heavily inspired by David Goldberg's classic paper, *What Every Computer Scientist Should Know About Floating-Point Arithmetic*. Through interactive visualisations, we will explore exactly how computers represent real numbers and the mathematical quirks that arise from these finite representations.

    If you've ever typed `0.1 + 0.2` into a programming language and gotten `0.30000000000000004`, you've encountered these quirks firsthand. Computers use a standard called **IEEE 754** to squeeze infinitely complex real numbers into a fixed number of binary bits (1s and 0s). 
    
    This process is essentially scientific notation for computers:

    $$ \\text{Number} = (-1)^{\\text{Sign}} \\times 1.\\text{Mantissa} \\times 2^{\\text{Exponent}} $$
    
    ---
    ### 📑 Table of Contents
    1. **[Floating Representations & The Spacing Subtlety](#1-floating-representations)**
    2. **[Rounding Error: ULPs vs. Relative Error](#2-rounding-error-ulps-vs-relative-error)**
    3. **[Guard Digits: Protecting the Math](#3-guard-digits)**
    4. **[Catastrophic Cancellation (Interactive)](#4-catastrophic-cancellation)**
    5. **[Modern AI Formats (NVIDIA A100)](#5-modern-ai-formats)**

    ---
    """)
    return


@app.cell
def sec1_floating_representations(mo):
    mo.md("""
    <a id="1-floating-representations"></a>
    ## 1. Floating Representations & The Spacing Subtlety

    If you've ever typed `0.1 + 0.2` into a Python console and gotten `0.30000000000000004`, you've encountered floating-point arithmetic. 

    Unlike integers, which have a constant gap of exactly `1` between every number on the number line, real numbers are infinite. To squeeze infinitely complex real numbers into a fixed 32 or 64 bits, computers use **IEEE 754** standard notation, which is essentially scientific notation for binary:

    $$ \\text{Number} = (-1)^{\\text{Sign}} \\times 1.\\text{Mantissa} \\times 2^{\\text{Exponent}} $$

    

    ### The "Stretchy" Number Line (Gaps and Scaling)
    The most crucial subtlety of floating-point numbers is that **the gap between numbers is not constant.** * Between $1.0$ and $2.0$, the gaps are incredibly tiny (high precision).
    * Between $1,000,000.0$ and $2,000,000.0$, the gaps are much larger.
    
    Every time the exponent increases by 1 (crossing a power of 2), the "ruler" stretches, and **the distance to the next representable number doubles.** This is why adding a very small number to a very large number often results in the small number being completely swallowed by the gap!
    
    
    """)
    return


@app.cell
def sec1_anatomy(mo):
    mo.md("""
    ### 🔬 The Anatomy of a Float

    Every floating-point number is broken down into specific components:
    * 🟥 **Sign Bit:** 1 bit for positive (`0`) or negative (`1`).
    * 🟩 **Exponent:** Determines the scale (how far the decimal point "floats"). It is stored with a bias to handle negative powers.
    * ⬜ **Implicit Bit:** Because binary scientific notation *always* starts with "1.", we don't store it in memory. It is assumed, giving us one extra bit of free precision!
    * 🟦 **Mantissa (Significand):** The actual fractional digits of the number.
    
    *(Hover your mouse over the bits below to instantly see their exact decimal values!)*
    """)
    return


@app.cell
def pythonic_fp_engine(Decimal, dataclass, struct):
    # --- Fluent Python Design Pattern: Data Models & Protocols ---
    
    @dataclass(frozen=True)
    class FPFormatConfig:
        """Data model configuring the rules for a specific FP format."""
        name: str
        desc: str
        struct_char: str       # 'd' for 64-bit, 'f' for 32-bit, 'e' for 16-bit
        total_bits: int
        exp_bits: int
        mantissa_bits: int
        exp_bias: int          # The bias subtracted from the exponent bits
        truncate_to: int | None = None # Used for AI formats that chop off bits

    class FloatParser:
        """
        Object-oriented wrapper that parses a float into bits and 
        utilizes the _repr_html_ protocol to intrinsically render itself.
        """
        def __init__(self, value: float, config: FPFormatConfig):
            self.value = value
            self.config = config
            self.sign, self.exponent, self.mantissa = self._extract_bits()
            self.stored_value = self._calculate_stored_value()
            self.error = abs(Decimal(str(self.value)) - Decimal(str(self.stored_value)))

        def _extract_bits(self) -> tuple[str, str, str]:
            try:
                b_bytes = struct.pack(f'>{self.config.struct_char}', self.value)
                bits = ''.join(f'{b:08b}' for b in b_bytes)
            except OverflowError:
                # Handle standard Infinity overflow for FP16
                bits = '0' + '1' * self.config.exp_bits + '0' * self.config.mantissa_bits

            if self.config.truncate_to:
                bits = bits[:self.config.truncate_to]
            
            s = bits[0]
            e = bits[1:1 + self.config.exp_bits]
            m = bits[1 + self.config.exp_bits:]
            return s, e, m

        def _calculate_stored_value(self) -> float:
            if not self.config.truncate_to:
                try:
                    return struct.unpack(f'>{self.config.struct_char}', struct.pack(f'>{self.config.struct_char}', self.value))[0]
                except OverflowError:
                    return float('inf')
            else:
                b_bytes = struct.pack(f'>{self.config.struct_char}', self.value)
                bits = ''.join(f'{b:08b}' for b in b_bytes)
                trunc_bits = bits[:self.config.truncate_to] + '0' * (self.config.total_bits - self.config.truncate_to)
                b_trunc = int(trunc_bits, 2).to_bytes(self.config.total_bits // 8, byteorder='big')
                return struct.unpack(f'>{self.config.struct_char}', b_trunc)[0]

        def _repr_html_(self) -> str:
            # Custom Instant CSS Tooltip Styling
            style_block = """
            <style>
            .fp-bit { position: relative; display: inline-block; }
            .fp-bit .fp-tip {
                visibility: hidden; background-color: #333; color: #fff; text-align: left;
                padding: 6px 10px; border-radius: 4px; position: absolute; z-index: 9999;
                bottom: 135%; left: 50%; transform: translateX(-50%); opacity: 0;
                transition: opacity 0.15s; font-size: 11px; white-space: nowrap; line-height: 1.4;
                pointer-events: none; font-family: sans-serif; font-weight: normal;
                box-shadow: 0px 4px 6px rgba(0,0,0,0.3);
            }
            .fp-bit .fp-tip::after {
                content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px;
                border-width: 5px; border-style: solid; border-color: #333 transparent transparent transparent;
            }
            .fp-bit:hover .fp-tip { visibility: visible; opacity: 1; }
            </style>
            """

            def draw_box(bit_str: str, role: str) -> str:
                styles = {
                    "sign": "background:#ffebee; color:#c62828; border:1px solid #ef9a9a;",
                    "exponent": "background:#e8f5e9; color:#2e7d32; border:1px solid #a5d6a7;",
                    "mantissa": "background:#e3f2fd; color:#1565c0; border:1px solid #90caf9;",
                    "implicit": "background:#f5f5f5; color:#9e9e9e; border:1px dashed #bdbdbd; opacity:0.8;"
                }
                
                boxes = []
                for i, bit in enumerate(bit_str):
                    # Construct rich HTML Tooltip Text based on the bit's role
                    if role == "mantissa":
                        power = -(i + 1)
                        dec_val = 2 ** power
                        contribution = dec_val if bit == '1' else 0.0
                        tooltip_text = f"<b>Position:</b> 2<sup>{power}</sup><br><b>Value:</b> {dec_val:.10g}<br><b>Adds:</b> <span style='color:#a5d6a7'>+{contribution:.10g}</span>"
                    elif role == "exponent":
                        power = len(bit_str) - 1 - i
                        dec_val = 2 ** power
                        contribution = dec_val if bit == '1' else 0
                        tooltip_text = f"<b>Position:</b> 2<sup>{power}</sup><br><b>Raw Adds:</b> <span style='color:#a5d6a7'>+{contribution}</span>"
                    elif role == "sign":
                        sign_val = "+1" if bit == '0' else "-1"
                        tooltip_text = f"<b>Sign Bit</b><br>Multiplier: <span style='color:#ef9a9a'>{sign_val}</span>"
                    elif role == "implicit":
                        tooltip_text = f"<b>Implicit Bit</b><br>Assumed: {bit}"
                    
                    tooltip_html = f"<span class='fp-tip'>{tooltip_text}</span>"
                    
                    boxes.append(
                        f"<div class='fp-bit' style='width:14px; height:20px; "
                        f"line-height:20px; text-align:center; font-family:monospace; "
                        f"font-size:12px; font-weight:bold; border-radius:3px; margin:1px; cursor:help; "
                        f"{styles.get(role, '')}'>{bit}{tooltip_html}</div>"
                    )
                return "".join(boxes)

            # Evaluate the mathematical components
            sign_int = int(self.sign)
            exp_int = int(self.exponent, 2)
            mantissa_fraction = sum(int(bit) * (2 ** -(i + 1)) for i, bit in enumerate(self.mantissa))
            
            # Determine the implicit bit based on whether it is a normal or subnormal number
            implicit_bit = "0" if exp_int == 0 else "1"
            
            if exp_int == 0:
                if mantissa_fraction == 0:
                    math_str = f"(-1)<sup>{sign_int}</sup> &times; 0.0 &times; 2<sup>0</sup>"
                else:
                    true_exp = 1 - self.config.exp_bias
                    math_str = f"(-1)<sup>{sign_int}</sup> &times; {mantissa_fraction} &times; 2<sup>{true_exp}</sup> &nbsp;<span style='font-size: 0.8em; color: #888;'>(Subnormal)</span>"
            elif exp_int == (1 << self.config.exp_bits) - 1:
                if mantissa_fraction == 0:
                    math_str = f"{'+&infin;' if sign_int == 0 else '-&infin;'}"
                else:
                    math_str = "NaN"
            else:
                true_exp = exp_int - self.config.exp_bias
                significand = 1.0 + mantissa_fraction
                math_str = f"(-1)<sup>{sign_int}</sup> &times; {significand:.6g} &times; 2<sup>{true_exp}</sup>"

            # Clean Flexbox Dashboard
            html_elements = [
                style_block,
                "<div style='display: flex; flex-direction: column; gap: 8px; width: 100%;'>",
                "<div>",
                draw_box(self.sign, "sign"),
                "<span style='margin:0 4px; color:#bbb;'>|</span>",
                draw_box(self.exponent, "exponent"),
                "<span style='margin:0 4px; color:#bbb;'>|</span>",
                draw_box(implicit_bit, "implicit") + "<span style='font-weight:bold; margin: 0 2px;'>.</span>",
                draw_box(self.mantissa, "mantissa"),
                "</div>",
                "<div style='display: flex; gap: 20px; font-size: 13px; color: #555; border-top: 1px dashed #ddd; padding-top: 8px;'>",
                "<div style='flex: 1;'>",
                f"<b>Stored Value:</b> {self.stored_value:.10g}... <br>",
                f"<b>Rounding Error:</b> ~{self.error:.2e}",
                "</div>",
                "<div style='flex: 1; border-left: 1px dashed #eee; padding-left: 15px; display: flex; align-items: center;'>",
                f"<span style='font-size: 14px; color: #222; font-family: \"Times New Roman\", Times, serif;'>",
                f"Value = {math_str}",
                "</span>",
                "</div>",
                "</div></div>"
            ]
            return "".join(html_elements)

    return FPFormatConfig, FloatParser


@app.cell
def define_configs(FPFormatConfig):
    SUPPORTED_FORMATS = [
        FPFormatConfig("FP64 (Double)", "Scientific computing standard. Huge range, massive precision.", 'd', 64, 11, 52, 1023),
        FPFormatConfig("FP32 (Single)", "Standard for 3D graphics and base machine learning.", 'f', 32, 8, 23, 127),
        FPFormatConfig("TF32 (TensorFloat)", "NVIDIA A100 AI format: Range of FP32, precision of FP16.", 'f', 32, 8, 10, 127, truncate_to=19),
        FPFormatConfig("BF16 (Bfloat16)", "Brain Float: Chops FP32 in half. Prevents AI overflow.", 'f', 32, 8, 7, 127, truncate_to=16),
        FPFormatConfig("FP16 (Half)", "Small exponent, easily overflows, very fast for inference.", 'e', 16, 5, 10, 15),
    ]
    return SUPPORTED_FORMATS,


@app.cell
def create_explorer_input(mo):
    number_input = mo.ui.number(value=0.1, step=0.1, label="**Enter a Real Number:**")
    return number_input,


@app.cell
def display_explorer_input(mo, number_input):
    mo.vstack([
        mo.md("#### 🧪 Interactive Explorer\nType `0.1`, `3.14159`, or `42` below to instantly see how the computer slices it up."),
        number_input
    ])


@app.cell
def display_reactive_grid(FloatParser, SUPPORTED_FORMATS, mo, number_input):
    try:
        current_val = float(number_input.value)
        is_valid = True
    except (ValueError, TypeError):
        is_valid = False
        current_val = 0.0

    if not is_valid:
        grid_ui = mo.md("⚠️ **Please enter a valid decimal number.**")
    else:
        rows = []
        for config in SUPPORTED_FORMATS:
            fp_object = FloatParser(current_val, config)
            row = mo.hstack([
                mo.md(f"**{config.name}**<br><span style='font-size:12px; color:#666;'>{config.desc}</span>"),
                fp_object
            ], widths=[1, 2.5], align="center")
            rows.append(row)
            rows.append(mo.Html("<hr style='margin: 10px 0; border: 0; border-top: 1px solid #eee;'>"))
        grid_ui = mo.vstack(rows)
        
    grid_ui


@app.cell
def sec2_rounding_errors(mo):
    mo.md("""
    <a id="2-rounding-error-ulps-vs-relative-error"></a>
    ## 2. Rounding Error: ULPs vs. Relative Error

    Because numbers like `0.1` create infinite repeating patterns in binary, they must be truncated and rounded to fit in memory. Measuring this error is tricky due to the "stretchy" number line. There are two main ways to measure it:

    ### ULP (Units in the Last Place)
    An ULP represents the physical gap between two adjacent floating-point numbers. If your error is "0.5 ULPs," it means your approximation is exactly halfway to the next tick mark on the ruler. 
    * **Intuition:** ULP is a measure of absolute error, but scaled by your local neighborhood.
    * **Use Case:** ULPs are perfect for measuring the accuracy of basic hardware operations (like a single addition or a square root). A perfectly rounded system guarantees a maximum error of 0.5 ULPs.

    ### Relative Error ($\epsilon$)
    Relative error is the absolute error divided by the true value. It measures the "noise-to-signal" ratio.
    * **Intuition:** Think of it like a zoom lens. It tells you the percentage of error regardless of the grid spacing.
    * **Use Case:** Relative error is vastly superior when establishing mathematical **error bounds** for long algorithms (like multiplying matrices).

    

    ### The "Wobble" Property
    Why is relative error better for bounds? If you take a number and multiply it by 2, you stretch it across the number line. 
    * If you measure with **ULPs**, jumping across an exponent boundary doubles the ULP gap size, making your error suddenly look twice as bad!
    * **Relative Error** gracefully ignores this boundary. It tracks the true proportion of error. The text by Goldberg notes that relative error "wobbles" (fluctuates smoothly between bounds) without violently snapping like ULPs do when crossing exponents.
    """)
    return


@app.cell
def sec3_guard_digits(mo):
    mo.md("""
    <a id="3-guard-digits"></a>
    ## 3. Guard Digits: Protecting the Math

    When adding or subtracting two floating-point numbers, the hardware must first shift the smaller number's mantissa to the right so its exponent matches the larger number. 

    If it shifts too far, bits fall off the end of the hardware register and vanish into the void. This causes massive, unnecessary errors.

    **The Solution: Guard Digits**
    To prevent this, ALUs (Arithmetic Logic Units) hold onto a few extra hidden bits—called **Guard Digits**—during the intermediate calculation, only rounding off at the very final step.

    Goldberg proves this with a famous theorem:
    > *If $x$ and $y$ are positive floating-point numbers in a format with base $\\beta$ and precision $p$, and if $(y/2) \\le x \\le 2y$, then $x - y$ is computed **exactly** if the arithmetic uses a single guard digit.*
    """)
    return


@app.cell
def sec4_cancellation(mo):
    mo.md("""
    <a id="4-catastrophic-cancellation"></a>
    ## 4. Catastrophic Cancellation (Interactive)

    Even with guard digits, subtraction poses the greatest threat to floating-point math. This is known as **Catastrophic Cancellation**.

    When you subtract two numbers that are very close to each other, the most significant digits match perfectly and cancel out to `0`. What is left behind? Only the least significant digits—the exact digits that are most heavily contaminated by accumulated rounding noise.

    
    
    ### Try it: $x^2 - y^2$ vs $(x - y)(x + y)$
    Algebraically, these formulas are identical. In computer science, they are vastly different.
    """)
    return


@app.cell
def create_catastrophic_inputs(mo):
    x_input = mo.ui.number(value=10000.1, step=0.1, label="**Value for X:**")
    y_input = mo.ui.number(value=10000.0, step=0.1, label="**Value for Y:**")
    return x_input, y_input


@app.cell
def display_catastrophic_inputs(mo, x_input, y_input):
    mo.hstack([x_input, y_input])


@app.cell
def calculate_and_display_cancellation(mo, np, x_input, y_input):
    x_val = x_input.value
    y_val = y_input.value
    
    # Force single precision logic
    x_fp32 = np.float32(x_val)
    y_fp32 = np.float32(y_val)

    # Method 1: The Naive way (Catastrophic Cancellation)
    x_sq = x_fp32 * x_fp32
    y_sq = y_fp32 * y_fp32
    result_bad = x_sq - y_sq

    # Method 2: The Safe way (Benign Cancellation)
    diff = x_fp32 - y_fp32
    summ = x_fp32 + y_fp32
    result_good = diff * summ

    # Exact True Value (using FP64 reference)
    true_val = (np.float64(x_val)**2) - (np.float64(y_val)**2)

    mo.md(f"""
    **The Math (in Single Precision FP32):**
    * $X$ is stored as: `{x_fp32:.7f}`
    * $Y$ is stored as: `{y_fp32:.7f}`

    **Method 1: Evaluating $X^2 - Y^2$ (The Bad Way)**
    * $X^2$ rounds to: `{x_sq}`
    * $Y^2$ rounds to: `{y_sq}`
    * Result ($X^2 - Y^2$): <span style='color:red; font-size: 1.2em; font-weight:bold;'>{result_bad}</span> ❌ *(The lower decimals got wiped out!)*

    **Method 2: Evaluating $(X - Y)(X + Y)$ (The Good Way)**
    * $(X - Y)$ evaluates cleanly to: `{diff}`
    * $(X + Y)$ evaluates to: `{summ}`
    * Result: <span style='color:green; font-size: 1.2em; font-weight:bold;'>{result_good}</span> ✅ *(Precision is preserved!)*

    *(For reference, the exact true math is: `{true_val}`)*
    """)


@app.cell
def sec5_a100(mo):
    mo.md("""
    <a id="5-modern-ai-formats"></a>
    ## 5. Modern AI Formats (NVIDIA A100 Tensor Cores)
    
    If you look at the Representation Explorer at the top of this page, you will notice three 16-to-32-bit formats: **FP32**, **TF32**, and **BF16**.

    

    When training neural networks, hardware architects realized two things:
    1. **Deep Learning needs a large range:** If the Exponent (🟩) is too small (like in standard **FP16**), gradients vanish or explode to `Infinity`.
    2. **Deep Learning doesn't need perfect precision:** The neural network acts like a noisy biological brain; it can tolerate slight errors in the Mantissa (🟦).
    
    **The A100 Tensor Core Solution:**
    * **BF16 (Brain Float):** Literally takes standard FP32 and just chops off the last 16 bits of the Mantissa. It keeps the exact same 8-bit Exponent for range but uses less memory.
    * **TF32 (TensorFloat-32):** NVIDIA's special hybrid. It uses an 8-bit Exponent (like FP32) to prevent `Infinity` crashes, but provides a 10-bit Mantissa (like FP16) to retain a bit more precision than BF16. It mathematically runs on hardware as a 19-bit format, massively speeding up matrix multiplication!
    """)
    return


if __name__ == "__main__":
    app.run()