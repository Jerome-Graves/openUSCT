// Testbench for acq_top: a full FMC acquisition co-simulation.
//
// Starts the sequencer, and for each transmit the sequencer fires, this bench
// streams that transmit's received channel samples (from OpenUAP), lets the
// capture run, reads the frame back, and acknowledges. The assembled frames are
// checked against the system in Python.

`timescale 1ns / 1ps

module tb_acq_top;

    localparam int N_ELEM   = 8;
    localparam int N_CH     = 8;
    localparam int DEPTH    = 64;
    localparam int SAMPLE_W = 16;
    localparam int ADDR_W   = $clog2(DEPTH);
    localparam int LEN      = 48;
    localparam int TOTAL    = 48;

    logic clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    logic                       start;
    logic [ADDR_W-1:0]          acq_delay, acq_len;
    logic                       adc_valid;
    logic signed [SAMPLE_W-1:0] adc_data [0:N_CH-1];
    logic [ADDR_W-1:0]          rd_addr;
    logic [$clog2(N_CH)-1:0]    rd_ch;
    logic signed [SAMPLE_W-1:0] rd_data;
    logic                       frame_taken, frame_valid;
    logic [$clog2(N_ELEM)-1:0]  tx_element;
    logic                       tx_fire, capture_trigger, busy, done;

    acq_top #(.N_ELEM(N_ELEM), .N_CH(N_CH), .DEPTH(DEPTH), .SAMPLE_W(SAMPLE_W))
        dut (.clk, .rst_n, .start, .acq_delay, .acq_len, .adc_valid, .adc_data,
             .rd_addr, .rd_ch, .rd_data, .frame_taken, .frame_valid,
             .tx_element, .tx_fire, .capture_trigger, .busy, .done);

    logic signed [SAMPLE_W-1:0] smem [0:N_ELEM*N_CH*TOTAL-1];
    integer fout, f, t, ch;

    initial begin
        $readmemh("acq_samples.hex", smem);
        fout = $fopen("acq_out.txt", "w");

        start = 0; adc_valid = 0; frame_taken = 0;
        acq_delay = '0; acq_len = LEN[ADDR_W-1:0]; rd_addr = '0; rd_ch = '0;
        for (ch = 0; ch < N_CH; ch++) adc_data[ch] = '0;
        repeat (3) @(posedge clk); rst_n = 1; @(posedge clk);
        start = 1; @(posedge clk); start = 0;

        for (f = 0; f < N_ELEM; f++) begin
            // Wait until the sequencer arms the capture for this transmit, then
            // one more clock so the capture datapath is in its CAP state.
            while (!capture_trigger) @(posedge clk);
            @(posedge clk);

            // Stream this transmit's received channel samples.
            for (t = 0; t < TOTAL; t++) begin
                adc_valid = 1;
                for (ch = 0; ch < N_CH; ch++)
                    adc_data[ch] = smem[f*N_CH*TOTAL + ch*TOTAL + t];
                @(posedge clk);
            end
            adc_valid = 0;

            // Wait for the captured frame, read it out, acknowledge.
            while (!frame_valid) @(posedge clk);
            for (ch = 0; ch < N_CH; ch++) begin
                for (t = 0; t < LEN; t++) begin
                    rd_ch = ch[$clog2(N_CH)-1:0];
                    rd_addr = t[ADDR_W-1:0];
                    @(posedge clk); @(posedge clk);
                    $fdisplay(fout, "%0d", rd_data);
                end
            end
            @(posedge clk); frame_taken = 1; @(posedge clk); frame_taken = 0;
        end

        while (!done) @(posedge clk);
        $fclose(fout);
        $finish;
    end

endmodule
