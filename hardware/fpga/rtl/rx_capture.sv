// rx_capture: multi-channel receive capture datapath.
//
// The core FPGA block of a full-matrix-capture acquisition system (no
// beamforming). On a transmit trigger it waits an acquisition delay, then
// captures a fixed-length window of samples from every receive channel in
// parallel into per-channel frame buffers. The captured frame, together with
// its acquisition delay, is exactly one UARP/UDSP Frame.
//
// The received waveforms are produced by the OpenUAP forward solver (the analog
// domain), quantised, and streamed in as ADC samples, so this is verified
// against the system it will run in.

`timescale 1ns / 1ps

module rx_capture #(
    parameter int N_CH     = 8,
    parameter int DEPTH    = 64,             // max samples per channel
    parameter int SAMPLE_W = 16,
    parameter int ADDR_W   = $clog2(DEPTH)
) (
    input  logic                        clk,
    input  logic                        rst_n,

    // Acquisition control.
    input  logic                        trigger,
    input  logic [ADDR_W-1:0]           acq_delay,   // samples to skip after trigger
    input  logic [ADDR_W-1:0]           acq_len,     // samples to capture

    // Parallel ADC input: one sample per channel per valid cycle.
    input  logic                        adc_valid,
    input  logic signed [SAMPLE_W-1:0]  adc_data [0:N_CH-1],

    output logic                        done,

    // Frame readout port.
    input  logic [ADDR_W-1:0]           rd_addr,
    input  logic [$clog2(N_CH)-1:0]     rd_ch,
    output logic signed [SAMPLE_W-1:0]  rd_data
);

    logic signed [SAMPLE_W-1:0] frame_mem [0:N_CH-1][0:DEPTH-1];

    typedef enum logic [1:0] {IDLE, SKIP, CAP} state_t;
    state_t state;
    logic [ADDR_W:0] cnt;   // one bit wider than an address
    integer ch;

    // Registered readout.
    always_ff @(posedge clk) rd_data <= frame_mem[rd_ch][rd_addr];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; cnt <= '0; done <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: if (trigger) begin
                    cnt   <= '0;
                    state <= (acq_delay == 0) ? CAP : SKIP;
                end
                SKIP: if (adc_valid) begin
                    if (cnt == acq_delay - 1) begin cnt <= '0; state <= CAP; end
                    else cnt <= cnt + 1'b1;
                end
                CAP: if (adc_valid) begin
                    for (ch = 0; ch < N_CH; ch++)
                        frame_mem[ch][cnt[ADDR_W-1:0]] <= adc_data[ch];
                    if (cnt == acq_len - 1) begin state <= IDLE; done <= 1'b1; end
                    else cnt <= cnt + 1'b1;
                end
                default: state <= IDLE;
            endcase
        end
    end

endmodule
