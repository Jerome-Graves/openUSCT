// Testbench for rx_capture.
//
// Streams per-channel ADC samples (written by the Python co-sim harness from
// OpenUAP-simulated waveforms) into the capture module for several transmit
// frames, each with its own acquisition delay, then reads back the captured
// frames to cap_out.txt for bit-exact checking against the system.

`timescale 1ns / 1ps

module tb_rx_capture;

    localparam int N_CH     = 8;
    localparam int DEPTH    = 64;
    localparam int SAMPLE_W = 16;
    localparam int ADDR_W   = $clog2(DEPTH);
    localparam int LEN      = 48;    // acquisition window
    localparam int TOTAL    = 64;    // samples streamed per frame
    localparam int NFRAMES  = 4;

    logic clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    logic                       trigger;
    logic [ADDR_W-1:0]          acq_delay, acq_len;
    logic                       adc_valid;
    logic signed [SAMPLE_W-1:0] adc_data [0:N_CH-1];
    logic                       done;
    logic [ADDR_W-1:0]          rd_addr;
    logic [$clog2(N_CH)-1:0]    rd_ch;
    logic signed [SAMPLE_W-1:0] rd_data;

    rx_capture #(.N_CH(N_CH), .DEPTH(DEPTH), .SAMPLE_W(SAMPLE_W))
        dut (.clk, .rst_n, .trigger, .acq_delay, .acq_len, .adc_valid, .adc_data,
             .done, .rd_addr, .rd_ch, .rd_data);

    logic signed [SAMPLE_W-1:0] smem [0:NFRAMES*N_CH*TOTAL-1];
    logic [ADDR_W-1:0]          dmem [0:NFRAMES-1];

    integer fout, f, t, ch;

    initial begin
        $readmemh("cap_samples.hex", smem);
        $readmemh("cap_delays.hex",  dmem);
        fout = $fopen("cap_out.txt", "w");

        trigger = 0; adc_valid = 0; acq_len = LEN[ADDR_W-1:0];
        for (ch = 0; ch < N_CH; ch++) adc_data[ch] = '0;
        repeat (3) @(posedge clk); rst_n = 1; @(posedge clk);

        for (f = 0; f < NFRAMES; f++) begin
            acq_delay = dmem[f];
            @(posedge clk); trigger = 1; @(posedge clk); trigger = 0;

            // Stream TOTAL samples for every channel (blocking, stable at edge).
            for (t = 0; t < TOTAL; t++) begin
                adc_valid = 1;
                for (ch = 0; ch < N_CH; ch++)
                    adc_data[ch] = smem[f*N_CH*TOTAL + ch*TOTAL + t];
                @(posedge clk);
            end
            adc_valid = 0;

            // Read back the captured window for every channel.
            for (ch = 0; ch < N_CH; ch++) begin
                for (t = 0; t < LEN; t++) begin
                    rd_ch   = ch[$clog2(N_CH)-1:0];
                    rd_addr = t[ADDR_W-1:0];
                    @(posedge clk); @(posedge clk);
                    $fdisplay(fout, "%0d", rd_data);
                end
            end
        end

        $fclose(fout);
        $finish;
    end

endmodule
