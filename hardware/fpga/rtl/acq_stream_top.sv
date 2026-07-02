// acq_stream_top: full acquisition subsystem with AXI-Stream frame output.
//
// tx_sequencer -> rx_capture -> axi_stream_out. Each transmit is fired, its
// received channels are captured into a frame, and the frame is streamed out
// over AXI-Stream to the host (which writes it into the UARP/UDSP file). The
// streamer's completion acknowledges the frame back to the sequencer, which
// then advances to the next transmit.

`timescale 1ns / 1ps

module acq_stream_top #(
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

    input  logic                        adc_valid,
    input  logic signed [SAMPLE_W-1:0]  adc_data [0:N_CH-1],

    // AXI-Stream master (to host DMA).
    output logic signed [SAMPLE_W-1:0]  m_tdata,
    output logic                        m_tvalid,
    output logic                        m_tlast,
    input  logic                        m_tready,

    output logic [CH_W-1:0]             tx_element,
    output logic                        tx_fire,
    output logic                        capture_trigger,
    output logic                        busy,
    output logic                        done
);

    logic cap_done, frame_valid, frame_taken;
    logic [ADDR_W-1:0] rd_addr;
    logic [$clog2(N_CH)-1:0] rd_ch;
    logic signed [SAMPLE_W-1:0] rd_data;

    // Start the streamer on the rising edge of frame_valid.
    logic fv_d;
    always_ff @(posedge clk or negedge rst_n)
        if (!rst_n) fv_d <= 1'b0; else fv_d <= frame_valid;
    wire stream_start = frame_valid & ~fv_d;

    tx_sequencer #(.N_ELEM(N_ELEM)) u_seq (
        .clk, .rst_n, .start,
        .capture_done(cap_done), .frame_taken(frame_taken),
        .tx_element, .tx_fire, .capture_trigger, .frame_valid, .busy, .done
    );

    rx_capture #(.N_CH(N_CH), .DEPTH(DEPTH), .SAMPLE_W(SAMPLE_W)) u_cap (
        .clk, .rst_n,
        .trigger(capture_trigger), .acq_delay, .acq_len,
        .adc_valid, .adc_data,
        .done(cap_done), .rd_addr, .rd_ch, .rd_data
    );

    axi_stream_out #(.N_CH(N_CH), .DEPTH(DEPTH), .SAMPLE_W(SAMPLE_W)) u_axis (
        .clk, .rst_n,
        .start(stream_start), .frame_len(acq_len),
        .rd_addr, .rd_ch, .rd_data,
        .m_tdata, .m_tvalid, .m_tlast, .m_tready,
        .done(frame_taken)
    );

endmodule
