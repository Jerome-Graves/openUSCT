// acq_top: full-matrix-capture acquisition subsystem.
//
// Connects the transmit sequencer to the receive capture datapath. On start it
// runs a complete FMC acquisition: for each element, fire it, capture the
// received channels into a frame, present the frame downstream, and advance.
// tx_element / tx_fire drive the external HV pulser; the ADC stream and frame
// readout are exposed for the host (or, in simulation, the testbench).

`timescale 1ns / 1ps

module acq_top #(
    parameter int N_ELEM   = 8,
    parameter int N_CH     = 8,
    parameter int DEPTH    = 64,
    parameter int SAMPLE_W = 16,
    parameter int ADDR_W   = $clog2(DEPTH),
    parameter int CH_W     = $clog2(N_ELEM)
) (
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        start,

    input  logic [ADDR_W-1:0]           acq_delay,
    input  logic [ADDR_W-1:0]           acq_len,

    // ADC stream in.
    input  logic                        adc_valid,
    input  logic signed [SAMPLE_W-1:0]  adc_data [0:N_CH-1],

    // Frame readout + downstream handshake.
    input  logic [ADDR_W-1:0]           rd_addr,
    input  logic [$clog2(N_CH)-1:0]     rd_ch,
    output logic signed [SAMPLE_W-1:0]  rd_data,
    input  logic                        frame_taken,
    output logic                        frame_valid,

    // Transmit control (to the pulser) and status.
    output logic [CH_W-1:0]             tx_element,
    output logic                        tx_fire,
    output logic                        capture_trigger,
    output logic                        busy,
    output logic                        done
);

    logic cap_done;

    tx_sequencer #(.N_ELEM(N_ELEM)) u_seq (
        .clk, .rst_n, .start,
        .capture_done(cap_done), .frame_taken,
        .tx_element, .tx_fire, .capture_trigger, .frame_valid, .busy, .done
    );

    rx_capture #(.N_CH(N_CH), .DEPTH(DEPTH), .SAMPLE_W(SAMPLE_W)) u_cap (
        .clk, .rst_n,
        .trigger(capture_trigger), .acq_delay, .acq_len,
        .adc_valid, .adc_data,
        .done(cap_done), .rd_addr, .rd_ch, .rd_data
    );

endmodule
