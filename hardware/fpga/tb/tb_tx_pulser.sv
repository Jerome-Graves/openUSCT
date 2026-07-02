// Testbench for tx_pulser: record the bipolar gate waveform for checking.

`timescale 1ns / 1ps

module tb_tx_pulser;

    localparam int PERIOD_W = 8;
    localparam int COUNT_W  = 6;
    localparam int HALF     = 8;
    localparam int NHC      = 6;
    localparam int DEAD     = 2;

    logic clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    logic                fire;
    logic [PERIOD_W-1:0] half_period, dead_time;
    logic [COUNT_W-1:0]  n_halfcycles;
    logic                pulse_p, pulse_n, busy, done;

    tx_pulser #(.PERIOD_W(PERIOD_W), .COUNT_W(COUNT_W)) dut (
        .clk, .rst_n, .fire, .half_period, .n_halfcycles, .dead_time,
        .pulse_p, .pulse_n, .busy, .done);

    integer fout, arg_half, arg_nhc, arg_dead;

    initial begin
        fout = $fopen("pulser_out.txt", "w");
        fire = 0;
        if (!$value$plusargs("HALF=%d", arg_half)) arg_half = HALF;
        if (!$value$plusargs("NHC=%d",  arg_nhc))  arg_nhc  = NHC;
        if (!$value$plusargs("DEAD=%d", arg_dead)) arg_dead = DEAD;
        half_period = arg_half[PERIOD_W-1:0];
        n_halfcycles = arg_nhc[COUNT_W-1:0];
        dead_time = arg_dead[PERIOD_W-1:0];
        repeat (3) @(posedge clk); rst_n = 1; @(posedge clk);

        fire = 1; @(posedge clk); fire = 0;

        while (!busy) @(posedge clk);
        while (busy) begin
            $fdisplay(fout, "%0d %0d", pulse_p, pulse_n);
            @(posedge clk);
        end

        $fclose(fout);
        $finish;
    end

endmodule
