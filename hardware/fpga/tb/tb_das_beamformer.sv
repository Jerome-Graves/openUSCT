// Testbench for das_beamformer.
//
// Loads per-channel samples, delays, and weights from hex files written by the
// Python co-simulation harness, runs one MAC pass per focal point, and writes
// the raw accumulator and rounded result to out.txt for bit-exact checking.

`timescale 1ns / 1ps

module tb_das_beamformer;

    localparam int N_CH     = 8;
    localparam int DEPTH    = 64;
    localparam int SAMPLE_W = 16;
    localparam int WEIGHT_W = 16;
    localparam int DELAY_W  = 10;
    localparam int ACC_W    = 40;
    localparam int SHIFT    = 15;
    localparam int NTRIALS  = 8;

    logic clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    logic                        load_en;
    logic [$clog2(N_CH)-1:0]     load_ch;
    logic [DELAY_W-1:0]          load_addr;
    logic signed [SAMPLE_W-1:0]  load_data;
    logic                        start;
    logic [DELAY_W-1:0]          delays  [0:N_CH-1];
    logic signed [WEIGHT_W-1:0]  weights [0:N_CH-1];
    logic signed [ACC_W-1:0]     acc_out, result;
    logic                        done;

    das_beamformer #(.N_CH(N_CH), .DEPTH(DEPTH), .SAMPLE_W(SAMPLE_W),
                     .WEIGHT_W(WEIGHT_W), .DELAY_W(DELAY_W), .ACC_W(ACC_W), .SHIFT(SHIFT))
        dut (.clk, .rst_n, .load_en, .load_ch, .load_addr, .load_data,
             .start, .delays, .weights, .acc_out, .result, .done);

    // Stimulus loaded from files.
    logic signed [SAMPLE_W-1:0] smem [0:N_CH*DEPTH-1];
    logic [DELAY_W-1:0]         dmem [0:NTRIALS*N_CH-1];
    logic signed [WEIGHT_W-1:0] wmem [0:NTRIALS*N_CH-1];

    integer fout, ch, addr, t;

    initial begin
        $readmemh("samples.hex", smem);
        $readmemh("delays.hex",  dmem);
        $readmemh("weights.hex", wmem);
        fout = $fopen("out.txt", "w");

        load_en = 0; start = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // Load per-channel sample buffers.
        for (ch = 0; ch < N_CH; ch++) begin
            for (addr = 0; addr < DEPTH; addr++) begin
                load_en   <= 1;
                load_ch   <= ch[$clog2(N_CH)-1:0];
                load_addr <= addr[DELAY_W-1:0];
                load_data <= smem[ch*DEPTH + addr];
                @(posedge clk);
            end
        end
        load_en <= 0;
        @(posedge clk);

        // One MAC pass per focal point.
        for (t = 0; t < NTRIALS; t++) begin
            for (ch = 0; ch < N_CH; ch++) begin
                delays[ch]  = dmem[t*N_CH + ch];
                weights[ch] = wmem[t*N_CH + ch];
            end
            start <= 1; @(posedge clk); start <= 0;
            wait (done);
            @(posedge clk);
            $fdisplay(fout, "%0d %0d", acc_out, result);
        end

        $fclose(fout);
        $finish;
    end

endmodule
