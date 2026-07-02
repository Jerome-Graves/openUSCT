// tx_pulser: bipolar transmit pulse-timing generator.
//
// On a fire trigger it drives the high-voltage pulser gate inputs to emit a
// bipolar square-wave burst: alternating positive and negative half-cycles at a
// programmable half-period (sets the centre frequency) for a programmable number
// of half-cycles. Each half-cycle begins with a dead-time window (both gates
// low) so the high-side and low-side devices are never on together (no
// shoot-through). This is the digital control for a pulser such as an
// STHV748 or an MD1210 + TC6320 bridge.

`timescale 1ns / 1ps

module tx_pulser #(
    parameter int PERIOD_W = 8,
    parameter int COUNT_W  = 6
) (
    input  logic                  clk,
    input  logic                  rst_n,
    input  logic                  fire,
    input  logic [PERIOD_W-1:0]   half_period,   // clocks per half-cycle
    input  logic [COUNT_W-1:0]    n_halfcycles,  // number of half-cycles
    input  logic [PERIOD_W-1:0]   dead_time,     // dead-time clocks per half-cycle

    output logic                  pulse_p,       // high-side gate
    output logic                  pulse_n,       // low-side gate
    output logic                  busy,
    output logic                  done
);

    typedef enum logic [1:0] {IDLE, RUN, FIN} state_t;
    state_t state;
    logic [COUNT_W-1:0]  hc;
    logic [PERIOD_W-1:0] cnt;

    wire hc_odd = hc[0];   // odd half-cycle -> negative polarity

    // Combinational gate outputs: active polarity only after the dead-time.
    always_comb begin
        pulse_p = 1'b0;
        pulse_n = 1'b0;
        if (state == RUN && cnt >= dead_time) begin
            pulse_p = ~hc_odd;   // even half-cycle: positive
            pulse_n =  hc_odd;   // odd half-cycle: negative
        end
    end

    assign busy = (state == RUN);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; hc <= '0; cnt <= '0; done <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: if (fire) begin hc <= '0; cnt <= '0; state <= RUN; end
                RUN: begin
                    if (cnt == half_period - 1) begin
                        cnt <= '0;
                        if (hc == n_halfcycles - 1) state <= FIN;
                        else hc <= hc + 1'b1;
                    end else cnt <= cnt + 1'b1;
                end
                FIN: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end

endmodule
