// tx_sequencer: full-matrix-capture transmit sequencing FSM.
//
// Drives a complete FMC acquisition: fire each element in turn, trigger the
// receive capture, wait for the frame, hand it off downstream, then advance to
// the next element. No beamforming; every element transmits once and all
// elements receive, producing an N-by-N multistatic dataset.

`timescale 1ns / 1ps

module tx_sequencer #(
    parameter int N_ELEM = 8,
    parameter int CH_W   = $clog2(N_ELEM)
) (
    input  logic             clk,
    input  logic             rst_n,
    input  logic             start,

    input  logic             capture_done,   // from rx_capture
    input  logic             frame_taken,    // downstream has read the frame

    output logic [CH_W-1:0]  tx_element,      // element to fire
    output logic             tx_fire,         // pulse: fire the current element
    output logic             capture_trigger, // pulse: start rx_capture
    output logic             frame_valid,     // a captured frame is ready
    output logic             busy,
    output logic             done
);

    typedef enum logic [2:0] {IDLE, FIRE, WAITCAP, READOUT, NEXTEL, FINISH} state_t;
    state_t state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; tx_element <= '0;
            tx_fire <= 1'b0; capture_trigger <= 1'b0; frame_valid <= 1'b0;
            busy <= 1'b0; done <= 1'b0;
        end else begin
            tx_fire <= 1'b0; capture_trigger <= 1'b0; done <= 1'b0;
            case (state)
                IDLE: begin
                    frame_valid <= 1'b0;
                    if (start) begin tx_element <= '0; busy <= 1'b1; state <= FIRE; end
                end
                FIRE: begin
                    tx_fire <= 1'b1; capture_trigger <= 1'b1;   // fire + arm capture
                    state <= WAITCAP;
                end
                WAITCAP: if (capture_done) begin
                    frame_valid <= 1'b1; state <= READOUT;
                end
                READOUT: if (frame_taken) begin
                    frame_valid <= 1'b0; state <= NEXTEL;
                end
                NEXTEL: if (tx_element == N_ELEM-1) state <= FINISH;
                        else begin tx_element <= tx_element + 1'b1; state <= FIRE; end
                FINISH: begin done <= 1'b1; busy <= 1'b0; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end

endmodule
