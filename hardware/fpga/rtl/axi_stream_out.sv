// axi_stream_out: stream a captured frame out as an AXI-Stream.
//
// On a start pulse it reads the per-channel frame buffer (via the rx_capture
// read port) and emits every sample as an AXI-Stream beat: channel-major order
// (channel outer, sample inner), tlast on the final beat. This is the interface
// a Zynq DMA reads, carrying frames to the host where they are written into the
// UARP/UDSP acquisition file. Honours tready backpressure.

`timescale 1ns / 1ps

module axi_stream_out #(
    parameter int N_CH     = 8,
    parameter int DEPTH    = 64,
    parameter int SAMPLE_W = 16,
    parameter int ADDR_W   = $clog2(DEPTH),
    parameter int CH_W     = $clog2(N_CH)
) (
    input  logic                        clk,
    input  logic                        rst_n,
    input  logic                        start,        // begin streaming one frame
    input  logic [ADDR_W-1:0]           frame_len,    // samples per channel

    // rx_capture read port.
    output logic [ADDR_W-1:0]           rd_addr,
    output logic [CH_W-1:0]             rd_ch,
    input  logic signed [SAMPLE_W-1:0]  rd_data,

    // AXI-Stream master.
    output logic signed [SAMPLE_W-1:0]  m_tdata,
    output logic                        m_tvalid,
    output logic                        m_tlast,
    input  logic                        m_tready,

    output logic                        done
);

    typedef enum logic [2:0] {IDLE, SETUP, RDWAIT, LATCH, SEND, FIN} state_t;
    state_t state;
    logic [CH_W-1:0]   ch;
    logic [ADDR_W-1:0] t;

    wire last_beat = (ch == N_CH-1) && (t == frame_len-1);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; ch <= '0; t <= '0;
            rd_addr <= '0; rd_ch <= '0;
            m_tdata <= '0; m_tvalid <= 1'b0; m_tlast <= 1'b0; done <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: begin
                    m_tvalid <= 1'b0;
                    if (start) begin ch <= '0; t <= '0; state <= SETUP; end
                end
                SETUP: begin
                    rd_ch <= ch; rd_addr <= t;   // address applies next cycle ...
                    state <= RDWAIT;
                end
                RDWAIT: state <= LATCH;          // ... and the read is registered
                LATCH: begin
                    m_tdata  <= rd_data;
                    m_tlast  <= last_beat;
                    m_tvalid <= 1'b1;
                    state    <= SEND;
                end
                SEND: if (m_tready) begin        // beat accepted
                    m_tvalid <= 1'b0;
                    if (last_beat) state <= FIN;
                    else begin
                        if (t == frame_len-1) begin t <= '0; ch <= ch + 1'b1; end
                        else t <= t + 1'b1;
                        state <= SETUP;
                    end
                end
                FIN: begin done <= 1'b1; state <= IDLE; end
                default: state <= IDLE;
            endcase
        end
    end

endmodule
