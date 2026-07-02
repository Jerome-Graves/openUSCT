# OpenUSCT Studio (GUI)

A pure-Python control and analysis application for OpenUSCT: the UARP-style
workflow (configure, acquire, image, reconstruct, export) with **no MATLAB
required**. It runs on the `ringfwi` stack and automatically uses the C++
backend when it is built.

```bash
pip install -r requirements.txt
streamlit run software/gui/app.py
```

Then open http://localhost:8501.

Choose **2D ring** or **3D cylinder** from the sidebar. In 3D the model, TFM
image, and reconstruction are shown as three orthogonal slices; 3D runs on a
modest grid and takes a little longer.

## Screens

- **Array & Transmit** — configure the array (2D ring or 3D cylinder), specimen,
  and flaw from the sidebar; see the model and the transmit wavelet and spectrum.
- **Acquisition** — run a simulated full-matrix-capture acquisition, view the
  received frames, and download the acquisition as a UARP/UDSP v4.0 file. A
  **Run FPGA capture co-sim** button feeds the received data through the real
  `rx_capture` SystemVerilog module in Icarus Verilog and confirms the RTL
  captures it bit-exact (needs `iverilog` on the path).
- **Imaging (TFM)** — total focusing method image of the acquisition.
- **Reconstruction (FWI)** — full waveform inversion, showing the recovered
  sound-speed map and the misfit convergence.

Everything the application computes uses the same functions as the rest of the
platform (and the same UARP format), so the GUI is a front end to the exact
pipeline that the tests and co-simulations verify.
