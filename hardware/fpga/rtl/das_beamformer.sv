// das_beamformer: fixed-point delay-and-sum receive beamformer.
//
// The core digital block of an array-ultrasound front end. Per-channel raw
// samples are written into an on-chip buffer. For each focal point the module
// is given a per-channel integer delay (a sample index) and an apodisation
// weight; it reads buffer[ch][delay[ch]], multiply-accumulates with the weight
// across all channels, and outputs the raw accumulator plus a rounded,
// down-shifted result.
//
// This mirrors the software delay-and-sum / total focusing method, so the
// Python imaging code is a ready-made golden model for bit-exact verification.

`timescale 1ns / 1ps

module das_beamformer #(
    parameter int N_CH     = 8,     // channels
    parameter int DEPTH    = 64,    // samples stored per channel
    parameter int SAMPLE_W = 16,    // sample bit width (signed)
    parameter int WEIGHT_W = 16,    // apodisation weight width (signed, Q1.15)
    parameter int DELAY_W  = 10,    // delay/address width
    parameter int ACC_W    = 40,    // accumulator width
    parameter int SHIFT    = 15     // output down-shift (Q1.15 weights)
) (
    input  logic                          clk,
    input  logic                          rst_n,

    // Sample load port.
    input  logic                          load_en,
    input  logic [$clog2(N_CH)-1:0]       load_ch,
    input  logic [DELAY_W-1:0]            load_addr,
    input  logic signed [SAMPLE_W-1:0]    load_data,

    // Per-focal-point compute.
    input  logic                          start,
    input  logic [DELAY_W-1:0]            delays  [0:N_CH-1],
    input  logic signed [WEIGHT_W-1:0]    weights [0:N_CH-1],

    output logic signed [ACC_W-1:0]       acc_out,   // raw MAC accumulator
    output logic signed [ACC_W-1:0]       result,    // rounded, down-shifted
    output logic                          done
);

    // Per-channel sample buffer.
    logic signed [SAMPLE_W-1:0] mem [0:N_CH-1][0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (load_en) mem[load_ch][load_addr] <= load_data;
    end

    typedef enum logic [1:0] {IDLE, MAC, FINISH} state_t;
    state_t state;
    logic [$clog2(N_CH+1)-1:0] ch;
    logic signed [ACC_W-1:0]   acc;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; ch <= '0; acc <= '0; done <= 1'b0;
            acc_out <= '0; result <= '0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: if (start) begin
                    acc <= '0; ch <= '0; state <= MAC;
                end
                MAC: begin
                    // Read delayed sample for this channel and multiply-accumulate.
                    acc <= acc + $signed(mem[ch][delays[ch]]) * $signed(weights[ch]);
                    if (ch == N_CH-1) state <= FINISH;
                    else ch <= ch + 1'b1;
                end
                FINISH: begin
                    acc_out <= acc;
                    // Symmetric rounding then arithmetic down-shift.
                    result  <= (acc + (SHIFT > 0 ? (40'sd1 <<< (SHIFT-1)) : 40'sd0)) >>> SHIFT;
                    done    <= 1'b1;
                    state   <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end

endmodule
