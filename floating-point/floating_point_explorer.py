import marimo

__generated_with = "0.1.0"
app = marimo.App(width="full")


@app.cell
def imports():
    import marimo as mo
    import struct
    from decimal import Decimal
    return mo, struct, Decimal


@app.cell
def header(mo):
    mo.md(
        """
        # 🧮 Demystifying Floating-Point Arithmetic
        
        Welcome! If you've ever typed `0.1 + 0.2` into a programming language and gotten `0.30000000000000004`, you've encountered the quirks of **floating-point arithmetic**. 
        
        Computers don't have infinite memory. They use a standard called **IEEE 754** to squeeze infinitely complex real numbers into a fixed number of binary bits (1s and 0s). This process is basically scientific notation for computers:
        
        $$ \text{Number} = (-1)^{\text{Sign}} \times 1.\text{Mantissa} \times 2^{\text{Exponent}} $$
        
        

        ### The Anatomy of a Float
        * 🟥 **Sign Bit:** 1 bit to say if the number is positive (0) or negative (1).
        * 🟩 **Exponent:** Determines where the "floating" decimal point sits (the scale).
        * ⬜ **Implicit Bit:** Because binary scientific notation *always* starts with "1.", we don't even store it! We just assume it's there to save space.
        * 🟦 **Mantissa (Significand):** The actual precision digits of the number.
        """
    )
    return


@app.cell
def interactive_input(mo):
    mo.md("### 🧪 Try it yourself! \nEnter a decimal number below. Try `0.1`, `3.14159`, or a whole number like `42` to see how the computer slices it up.")
    
    number_input = mo.ui.text(value="0.1", label="**Enter a Real Number:**")
    number_input
    return number_input,


@app.cell
def bit_helpers(struct):
    def bit_box(bit, role):
        """Creates color-coded, rounded HTML boxes for bits."""
        styles = {
            "sign": "background:#ffebee; color:#c62828; border:1px solid #ef9a9a;",
            "exponent": "background:#e8f5e9; color:#2e7d32; border:1px solid #a5d6a7;",
            "mantissa": "background:#e3f2fd; color:#1565c0; border:1px solid #90caf9;",
            "implicit": "background:#f5f5f5; color:#9e9e9e; border:1px dashed #bdbdbd; opacity:0.8;"
        }
        style = styles.get(role, "")
        return f"<div style='display:inline-block; width:14px; height:20px; line-height:20px; text-align:center; font-family:monospace; font-size:12px; font-weight:bold; border-radius:3px; margin:1px; {style}'>{bit}</div>"

    def float_to_bits(f, format_type):
        """Packs a float into bytes, then extracts binary string."""
        if format_type == "FP64":
            bits = ''.join(f'{b:08b}' for b in struct.pack('>d', f))
            return bits[0], bits[1:12], bits[12:]
        elif format_type in ["FP32", "BF16", "TF32"]:
            bits = ''.join(f'{b:08b}' for b in struct.pack('>f', f))
            if format_type == "FP32": return bits[0], bits[1:9], bits[9:]
            if format_type == "BF16": return bits[0], bits[1:9], bits[9:16]
            if format_type == "TF32": return bits[0], bits[1:9], bits[9:19]
        elif format_type == "FP16":
            try:
                bits = ''.join(f'{b:08b}' for b in struct.pack('>e', f))
                return bits[0], bits[1:6], bits[6:]
            except OverflowError:
                return "0", "11111", "0000000000" # Inf
    
    def decode_truncated(f, format_type):
        """Recalculates the exact decimal value of truncated formats."""
        if format_type == "FP64": return f
        b32 = struct.pack('>f', f)
        bits = ''.join(f'{b:08b}' for b in b32)
        if format_type == "FP32": return struct.unpack('>f', b32)[0]
        if format_type == "TF32":
            trunc_bits = bits[:19] + '0'*13
            b_trunc = int(trunc_bits, 2).to_bytes(4, byteorder='big')
            return struct.unpack('>f', b_trunc)[0]
        if format_type == "BF16":
            trunc_bits = bits[:16] + '0'*16
            b_trunc = int(trunc_bits, 2).to_bytes(4, byteorder='big')
            return struct.unpack('>f', b_trunc)[0]
        if format_type == "FP16":
            try: return struct.unpack('>e', struct.pack('>e', f))[0]
            except: return float('inf')

    return bit_box, float_to_bits, decode_truncated


@app.cell
def visualization_grid(mo, number_input, bit_box, float_to_bits, decode_truncated, Decimal):
    try:
        val = float(number_input.value)
        valid_input = True
    except ValueError:
        valid_input = False

    if valid_input:
        formats = [
            ("FP64 (Double Precision)", "FP64", "The gold standard for scientific computing. Huge range, massive precision."),
            ("FP32 (Single Precision)", "FP32", "Standard for most 3D graphics and basic machine learning."),
            ("TF32 (TensorFloat-32)", "TF32", "NVIDIA A100 AI format: Range of FP32, precision of FP16."),
            ("BF16 (Bfloat16)", "BF16", "Brain Float: Chops FP32 in half. Prevents overflow in deep learning."),
            ("FP16 (Half Precision)", "FP16", "Smaller exponent, easily overflows, but very fast for AI inference.")
        ]
        
        grid_rows = []
        for name, fmt, desc in formats:
            s, e, m = float_to_bits(val, fmt)
            stored_val = decode_truncated(val, fmt)
            
            # Format the bits visually
            html_out = [bit_box(s, "sign"), "<span style='margin:0 4px; color:#bbb;'>|</span>"]
            for b in e: html_out.append(bit_box(b, "exponent"))
            html_out.append("<span style='margin:0 4px; color:#bbb;'>|</span>")
            html_out.append(bit_box("1", "implicit") + "<span style='font-weight:bold;'>.</span>")
            for b in m: html_out.append(bit_box(b, "mantissa"))
            
            # Show rounding error details
            error = abs(Decimal(str(val)) - Decimal(str(stored_val)))
            error_text = f"<div style='font-size: 13px; color: #555; margin-top: 5px;'><b>Stored Value:</b> {stored_val:.10g}... <br><b>Rounding Error:</b> ~{error:.2e}</div>"
            
            grid_rows.append([
                mo.md(f"**{name}**<br><span style='font-size:12px; color:#666;'>{desc}</span>"),
                mo.Html("".join(html_out) + error_text)
            ])
            
        display_output = mo.ui.table(grid_rows, selection=None)
    else:
        display_output = mo.md("⚠️ **Please enter a valid decimal number.**")

    display_output
    return display_output, val, valid_input


@app.cell
def a100_explanation(mo):
    mo.md(
        """
        ---
        ## 🧠 Why did NVIDIA invent TF32 for the A100?
        
        Look closely at the grid above, specifically comparing **FP32**, **TF32**, and **BF16**. 
        
        

        When training neural networks, researchers realized two things:
        1. **Deep Learning needs a large range:** If the Exponent (🟩) is too small (like in standard **FP16**), gradients vanish or explode to `Infinity`. 
        2. **Deep Learning doesn't need perfect precision:** The neural network acts like a noisy biological brain; it can tolerate slight errors in the Mantissa (🟦).

        **The A100 Tensor Core Solution:**
        * **BF16 (Brain Float):** Literally takes standard FP32 and just chops off the last 16 bits of the Mantissa. It keeps the exact same 8-bit Exponent for range but uses less memory.
        * **TF32 (TensorFloat-32):** NVIDIA's special hybrid. It uses an 8-bit Exponent (like FP32) to prevent `Infinity` crashes, but provides a 10-bit Mantissa (like FP16) to retain a bit more precision than BF16. It mathematically runs on hardware as a 19-bit format, massively speeding up matrix multiplication!
        """
    )
    return

if __name__ == "__main__":
    app.run()
