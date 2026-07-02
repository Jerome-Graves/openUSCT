// Testbench for acq_stream_top: full acquisition with AXI-Stream frame output.
//
// Drives the acquisition (feeding each transmit's received samples) and, in
// parallel, collects the AXI-Stream frame beats (with tready backpressure) to
// stream_out.txt for checking against the system in Python.

`timescale 1ns / 1ps

module tb_acq_stream;

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
    logic signed [SAMPLE_W-1:0] m_tdata;
    logic                       m_tvalid, m_tlast, m_tready;
    logic [$clog2(N_ELEM)-1:0]  tx_element;
    logic                       tx_fire, capture_trigger, busy, done;

    acq_stream_top #(.N_ELEM(N_ELEM), .N_CH(N_CH), .DEPTH(DEPTH), .SAMPLE_W(SAMPLE_W))
        dut (.clk, .rst_n, .start, .acq_delay, .acq_len, .adc_valid, .adc_data,
             .m_tdata, .m_tvalid, .m_tlast, .m_tready,
             .tx_element, .tx_fire, .capture_trigger, .busy, .done);

    logic signed [SAMPLE_W-1:0] smem [0:N_ELEM*N_CH*TOTAL-1];
    integer fout, f, t, ch;

    // tready backpressure: deassert one cycle in four.
    logic [1:0] bp = 0;
    always_ff @(posedge clk) bp <= bp + 1'b1;
    assign m_tready = (bp != 2'd0);

    // Collect accepted AXI-Stream beats.
    initial fout = $fopen("stream_out.txt", "w");
    always_ff @(posedge clk)
        if (rst_n && m_tvalid && m_tready)
            $fdisplay(fout, "%0d %0d", m_tdata, m_tlast);

    initial begin
        $readmemh("acq_samples.hex", smem);
        start = 0; adc_valid = 0;
        acq_delay = '0; acq_len = LEN[ADDR_W-1:0];
        for (ch = 0; ch < N_CH; ch++) adc_data[ch] = '0;
        repeat (3) @(posedge clk); rst_n = 1; @(posedge clk);
        start = 1; @(posedge clk); start = 0;

        for (f = 0; f < N_ELEM; f++) begin
            while (!capture_trigger) @(posedge clk);
            @(posedge clk);   // let capture reach CAP
            for (t = 0; t < TOTAL; t++) begin
                adc_valid = 1;
                for (ch = 0; ch < N_CH; ch++)
                    adc_data[ch] = smem[f*N_CH*TOTAL + ch*TOTAL + t];
                @(posedge clk);
            end
            adc_valid = 0;
            // The frame streams out via AXIS (collected above); the sequencer
            // advances when the streamer finishes, re-asserting capture_trigger.
        end

        while (!done) @(posedge clk);
        repeat (10) @(posedge clk);
        $fclose(fout);
        $finish;
    end

endmodule
