// ??? key decision to make is the bit numbering of a word.
// For verilog it seems esier to think MSB(63) - LSB(0). Extract and deposit become easier too.

// ??? we write an emulator... not a simulator, since we can interpert all intructions.


module system (
    input clk,
    input reset
);
    wire [31:0] addr;
    wire [63:0] wdata, rdata;
    wire mem_read, mem_write;

    // CPU core
    cpu my_cpu (
        .clk(clk),
        .reset(reset),
        .addr(addr),
        .write_data(wdata),
        .read_data(rdata),
        .read_en(mem_read),
        .write_en(mem_write)
    );

    // Memory block
    memory mem (
        .clk(clk),
        .write_en(mem_write),
        .read_en(mem_read),
        .addr(addr),
        .write_data(wdata),
        .read_data(rdata)
    );
endmodule

module cpu (
    input wire clk,
    input wire reset,
    output reg [31:0] addr,
    output reg [63:0] write_data,
    input  wire [63:0] read_data,
    output reg read_en,
    output reg write_en,
    output reg [7:0] reg_a
);
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            addr <= 0;
            write_data <= 0;
            read_en <= 0;
            write_en <= 0;
            reg_a <= 0;
        end else begin
            // Example: increment reg_a, write to memory, then read back
            reg_a <= reg_a + 1;
            addr <= 32'h00000004;
            write_data <= {56'd0, reg_a};
            write_en <= 1;
            read_en <= 0;
        end
    end
endmodule


module memory (
    input wire clk,
    input wire write_en,
    input wire read_en,
    input wire [31:0] addr,
    input wire [63:0] write_data,
    output reg [63:0] read_data
);
    parameter MEM_WORDS = 1024;
    reg [63:0] mem [0:MEM_WORDS-1];
    wire [31:3] word_addr = addr[31:3];

    always @(posedge clk) begin
        if (read_en)
            read_data <= mem[word_addr];
        if (write_en)
            mem[word_addr] <= write_data;
    end

    // Debug access
    task debug_read(input [31:0] a);
        $display("MEM[%0d] = %h", a, mem[a]);
    endtask

    task debug_write(input [31:0] a, input [63:0] d);
        mem[a] = d;
        $display("WROTE MEM[%0d] = %h", a, d);
    endtask
endmodule



// reg file for general registers 2 read, 2 write ( expensive )

module regfile (
    
    input  logic        clk,
    input  logic        rst,            // Synchronous reset
    input  logic        we_a, we_b,      // Write enables for port A and B
    input  logic [3:0]  waddr_a, waddr_b,// Write addresses for port A and B
    input  logic [63:0] wdata_a, wdata_b,// Write data for port A and B
    input  logic [3:0]  raddr_0, raddr_1, raddr_2, raddr_3, // Read addresses
    output logic [63:0] rdata_0, rdata_1, rdata_2, rdata_3  // Read data outputs
);

    logic [63:0] regs [15:1]; // Registers 1 to 15 (reg[0] is hardwired to zero)

    // Write logic: A has priority over B, register 0 is always zero
    always_ff @(posedge clk) begin
        if (rst) begin
            for (int i = 1; i < 16; i++) begin
                regs[i] <= 64'd0;
            end
        end else begin
            if (we_b && waddr_b != 4'd0 && (!we_a || (waddr_a != waddr_b))) begin
                regs[waddr_b] <= wdata_b;
            end
            if (we_a && waddr_a != 4'd0) begin
                regs[waddr_a] <= wdata_a;
            end
        end
    end

    // Read logic (asynchronous reads)
    assign rdata_0 = (raddr_0 == 4'd0) ? 64'd0 : regs[raddr_0];
    assign rdata_1 = (raddr_1 == 4'd0) ? 64'd0 : regs[raddr_1];
    assign rdata_2 = (raddr_2 == 4'd0) ? 64'd0 : regs[raddr_2];
    assign rdata_3 = (raddr_3 == 4'd0) ? 64'd0 : regs[raddr_3];

endmodule

// reg file for control registers....

module spr_file (
    input  wire        clk,

    // Single indexed access
    input  wire        wr_en,
    input  wire [3:0]  wr_idx,
    input  wire [63:0] wr_data,
    input  wire [3:0]  rd_idx,
    output wire [63:0] rd_data,

    // Multi-write trap update
    input  wire        trap_wr_en,
    input  wire [63:0] trap_addr,
    input  wire [63:0] trap_inst,
    input  wire [63:0] trap_arg0,
    input  wire [63:0] trap_arg1,

    // Alias outputs
    output wire [63:0] spr_trap_addr,
    output wire [63:0] spr_trap_inst,
    output wire [63:0] spr_trap_arg0,
    output wire [63:0] spr_trap_arg1,

    // PID outputs
    output wire [19:0] spr_pid0,
    output wire [19:0] spr_pid1,
    output wire [19:0] spr_pid2,
    output wire [18:0] spr_pid3,
    output wire [19:0] spr_pid4,
    output wire [19:0] spr_pid5,
    output wire [19:0] spr_pid6,
    output wire [18:0] spr_pid7,

    // PID WD outputs
    output wire [19:0] spr_wd_pid0,
    output wire [19:0] spr_wd_pid1,
    output wire [19:0] spr_wd_pid2,
    output wire [18:0] spr_wd_pid3,
    output wire [19:0] spr_wd_pid4,
    output wire [19:0] spr_wd_pid5,
    output wire [19:0] spr_wd_pid6,
    output wire [18:0] spr_wd_pid7

);
    reg [63:0] spr [15:0];

    // Write port
    always @(posedge clk) begin
        if (wr_en) begin
            spr[wr_idx] <= wr_data;
        end

        if (trap_wr_en) begin
            spr[4] <= trap_addr;
            spr[5] <= trap_inst;
            spr[6] <= trap_arg0;
            spr[7] <= trap_arg1;
        end
    end

    // Read port
    assign rd_data = spr[rd_idx];

    // Aliases for trap group
    assign spr_trap_addr = spr[4];
    assign spr_trap_inst = spr[5];
    assign spr_trap_arg0 = spr[6];
    assign spr_trap_arg1 = spr[7];

    // Aliases for PID 
    assign spr_pid0 = spr[4][52:33]; 
    assign spr_wd_pid0 = spr[4][32];

    // andc so on .... fix numbers first ...

endmodule




module alu64 (

    input  logic [63:0] a,              // Operand A
    input  logic [63:0] b,              // Operand B
    input  logic [4:0]  op,             // ALU operation selector
    input  logic [1:0]  a_shift,        // Shift amount for A (00 = no shift, 01 = 1-bit, 10 = 2-bit, 11 = 3-bit)
    input  logic        b_invert,       // Invert B
    input  logic        out_invert,     // Invert output
    output logic [63:0] result,         // ALU result
    output logic trap                   // Trap signal (for overflow conditions)
);

    logic [63:0] a_shifted, b_modified, alu_out;
    logic signed [63:0] signed_a, signed_b, signed_result;
    logic shift_trap, add_sub_trap;

    // Apply left shift to operand A and check for trap condition
    always_comb begin
        case (a_shift)
            2'b00: begin 
                a_shifted = a; 
                shift_trap = 0; 
            end
            2'b01: begin 
                a_shifted = a << 1; 
                shift_trap = a[63]; // Traps if '1' is shifted out
            end
            2'b10: begin 
                a_shifted = a << 2; 
                shift_trap = a[63] | a[62]; // Traps if any '1' is shifted out
            end
            2'b11: begin 
                a_shifted = a << 3; 
                shift_trap = a[63] | a[62] | a[61]; // Traps if any '1' is shifted out
            end
        endcase
    end

    // Apply inversion to operand B if needed
    assign b_modified = b_invert ? ~b : b;

    // Signed versions for comparison
    assign signed_a = a_shifted;
    assign signed_b = b_modified;

    // Default trap signals
    assign trap = shift_trap | add_sub_trap;

    always_comb begin
        case (op)
            5'b00000: begin // Signed Addition with Overflow
                signed_result = signed_a + signed_b;
                alu_out = signed_result;
                add_sub_trap = ((signed_a > 0 && signed_b > 0 && signed_result < 0) ||
                                (signed_a < 0 && signed_b < 0 && signed_result > 0));
            end

            5'b00001: begin // Signed Subtraction with Overflow
                signed_result = signed_a - signed_b;
                alu_out = signed_result;
                add_sub_trap = ((signed_a > 0 && signed_b < 0 && signed_result < 0) ||
                                (signed_a < 0 && signed_b > 0 && signed_result > 0));
            end

            5'b00010: alu_out = a_shifted & b_modified;  // AND
            5'b00011: alu_out = a_shifted | b_modified;  // OR
            5'b00100: alu_out = a_shifted ^ b_modified;  // XOR
            5'b00101: alu_out = a_shifted;               // Pass A
            5'b00110: alu_out = b_modified;              // Pass B
            5'b00111: alu_out = (a_shifted == b_modified) ? 64'd1 : 64'd0; // Equal
            5'b01000: alu_out = (a_shifted != b_modified) ? 64'd1 : 64'd0; // Not Equal
            5'b01001: alu_out = (signed_a < signed_b) ? 64'd1 : 64'd0; // Signed Less Than
            5'b01010: alu_out = (signed_a <= signed_b) ? 64'd1 : 64'd0; // Signed Less or Equal

            5'b01011: begin // Shift A and Add B (with overflow detection)
                signed_result = signed_a + signed_b;
                alu_out = signed_result;
                add_sub_trap = ((signed_a > 0 && signed_b > 0 && signed_result < 0) ||
                                (signed_a < 0 && signed_b < 0 && signed_result > 0));
            end

            default: alu_out = 64'd0; // Default case (NOP)
        endcase
    end

    // Apply output inversion if needed
    assign result = out_invert ? ~alu_out : alu_out;

endmodule


// needs to be adapted to our needs... just the general idea...

module SignExtender (
    input  [29:0] instr,        // 30-bit instruction
    input  [1:0]  field_sel,    // Selects which field to extract
    output reg [63:0] result    // 64-bit sign-extended result
);
    reg [9:0] field;            // Maximum field size (adjust if needed)

    always @(*) begin
        case (field_sel)
            2'b00: field = instr[14:5];  // Offset 5, Length 10
            2'b01: field = instr[17:15]; // Offset 15, Length 3
            // Add more cases as needed
            default: field = 0;
        endcase

        // Sign-extend the extracted field
        result = {{(64-$bits(field)){field[$bits(field)-1]}}, field};
    end
endmodule


module extract_field (
    input  [63:0] src,
    input  [5:0]  pos,         // Rightmost bit of the field (LSB)
    input  [5:0]  len,         // Length of the field
    input         sign_extend, // If set, perform sign extension
    output [63:0] result
);

    wire [63:0] raw = src[pos + len - 1 : pos];

    assign result = sign_extend && raw[len - 1] ?
                    {{(64 - len){1'b1}}, raw} :
                    {{(64 - len){1'b0}}, raw};

endmodule


module deposit_field (
    input  [63:0] target,      // Destination register
    input  [63:0] src,         // Source data (field in LSB)
    input  [5:0]  pos,         // Rightmost bit position in target (LSB)
    input  [5:0]  len,         // Length of the field to insert
    output [63:0] result
);

    wire [63:0] field = src[len - 1:0];  // Only take relevant bits
    wire [63:0] mask  = ~(((64'h1 << len) - 1) << pos); // Clear target bits
    wire [63:0] insert = field << pos;

    assign result = (target & mask) | insert;

endmodule

//  check module for PDI protection 

module match8 #(
    parameter WIDTH = 20
)(
    input  [WIDTH-1:0] a0,
    input  [WIDTH-1:0] a1,
    input  [WIDTH-1:0] a2,
    input  [WIDTH-1:0] a3,
    input  [WIDTH-1:0] a4,
    input  [WIDTH-1:0] a5,
    input  [WIDTH-1:0] a6,
    input  [WIDTH-1:0] a7,
    input  [WIDTH-1:0] x,
    output             y
);

    assign y = (x == a0) |
               (x == a1) |
               (x == a2) |
               (x == a3) |
               (x == a4) |
               (x == a5) |
               (x == a6) |
               (x == a7);

endmodule





// Two way associative cache


module two_way_associative_cache_async_read #(
    parameter CACHE_SIZE = 256, // Number of cache blocks
    parameter BLOCK_SIZE = 4,   // Size of each block in words
    parameter WORD_SIZE = 32    // Size of each word in bits
)(
    input wire clk,
    input wire reset,
    input wire [WORD_SIZE-1:0] address,
    input wire [WORD_SIZE-1:0] write_data,
    input wire write,
    output reg [WORD_SIZE-1:0] read_data,
    output reg hit
);

localparam NUM_WAYS = 2;
localparam NUM_SETS = CACHE_SIZE / (NUM_WAYS * BLOCK_SIZE);
localparam INDEX_BITS = $clog2(NUM_SETS);
localparam OFFSET_BITS = $clog2(BLOCK_SIZE);
localparam TAG_BITS = WORD_SIZE - INDEX_BITS - OFFSET_BITS;

reg [WORD_SIZE-1:0] cache_data[NUM_WAYS-1:0][NUM_SETS-1:0][BLOCK_SIZE-1:0];
reg [TAG_BITS-1:0] tag_array[NUM_WAYS-1:0][NUM_SETS-1:0];
reg valid_array[NUM_WAYS-1:0][NUM_SETS-1:0];
reg lru[NUM_SETS-1:0];

wire [TAG_BITS-1:0] tag = address[WORD_SIZE-1:WORD_SIZE-TAG_BITS];
wire [INDEX_BITS-1:0] index = address[WORD_SIZE-TAG_BITS-1:OFFSET_BITS];
wire [OFFSET_BITS-1:0] offset = address[OFFSET_BITS-1:0];

// Cache hit logic
wire way0_hit = (valid_array[0][index] && tag_array[0][index] == tag);
wire way1_hit = (valid_array[1][index] && tag_array[1][index] == tag);

// Combinational read logic for asynchronous read
always @(*) begin
    if (way0_hit) begin
        read_data = cache_data[0][index][offset];
        hit = 1;
    end else if (way1_hit) begin
        read_data = cache_data[1][index][offset];
        hit = 1;
    end else begin
        read_data = 0; // Return some default value on cache miss
        hit = 0;
    end
end


//  Cache with bypass


module cache_bypass (
  input wire clk,
  input wire reset,
  input wire [31:0] addr,
  input wire [31:0] data_in,
  input wire valid,
  input wire bypass,
  output wire [31:0] data_out,
  output wire hit,
  output wire ready
);
  parameter NUM_BLOCKS = 64;
  parameter TAG_WIDTH = 20;
  parameter ADDR_WIDTH = 32;
  parameter BLOCK_OFFSET = 6;  // Assuming block size of 64B
  
  reg [31:0] cache_mem [0:NUM_BLOCKS-1];
  reg [TAG_WIDTH-1:0] tag_array [0:NUM_BLOCKS-1];
  reg valid_array [0:NUM_BLOCKS-1];
  
  wire [TAG_WIDTH-1:0] tag = addr[ADDR_WIDTH-1:BLOCK_OFFSET];
  wire [BLOCK_OFFSET-1:0] index = addr[BLOCK_OFFSET-1:0];
  
  wire cache_hit = valid_array[index] && (tag_array[index] == tag);
  wire [31:0] mem_data;  // Assume this is connected to main memory module
  
  wire use_cache = ~bypass && cache_hit;
  assign data_out = use_cache ? cache_mem[index] : mem_data;
  assign hit = cache_hit;
  assign ready = (use_cache || bypass);  // Data ready in the same cycle
  
  // Cache update logic here (on cache miss, etc.)
  
endmodule



// Write logic (synchronous)
always @(posedge clk or posedge reset) begin
    if (reset) begin
        integer i, j;
        for (i = 0; i < NUM_WAYS; i = i + 1) begin
            for (j = 0; j < NUM_SETS; j = j + 1) begin
                valid_array[i][j] <= 0;
            end
        end
    end else if (write) begin
        if (way0_hit) begin
            cache_data[0][index][offset] <= write_data;
            lru[index] <= 1;
        end else if (way1_hit) begin
            cache_data[1][index][offset] <= write_data;
            lru[index] <= 0;
        end else begin
            // Cache miss, choose LRU way for replacement
            integer way = lru[index] ? 1 : 0;
            cache_data[way][index][offset] <= write_data;
            tag_array[way][index] <= tag;
            valid_array[way][index] <= 1;
            lru[index] <= ~lru[index];
        end
    end
end

endmodule



// Branch Target Buffer Notes

module btb #(
    parameter BTB_ENTRIES = 16,     // Number of BTB entries
    parameter ADDR_WIDTH = 32       // Width of addresses (PC and Target)
)(
    input wire clk,
    input wire reset,
    
    // Inputs
    input wire [ADDR_WIDTH-1:0] pc_in,         // Current program counter (PC)
    input wire branch_taken,                   // Whether the branch is taken
    input wire [ADDR_WIDTH-1:0] target_addr,   // Actual target address
    
    // Outputs
    output reg hit,                            // Hit or miss signal
    output reg [ADDR_WIDTH-1:0] predicted_target // Predicted target address
);
    
    // BTB storage: each entry has a PC and a target address
    reg [ADDR_WIDTH-1:0] btb_pc    [BTB_ENTRIES-1:0];   // Array of PCs
    reg [ADDR_WIDTH-1:0] btb_target[BTB_ENTRIES-1:0];   // Array of target addresses
    
    // Index calculation for direct-mapped BTB (simple modulo)
    wire [$clog2(BTB_ENTRIES)-1:0] index = pc_in[$clog2(BTB_ENTRIES)-1:0];
    
    integer i;
    
    // BTB lookup and update logic
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            // On reset, clear all BTB entries
            hit <= 0;
            predicted_target <= 0;
            for (i = 0; i < BTB_ENTRIES; i = i + 1) begin
                btb_pc[i] <= 0;
                btb_target[i] <= 0;
            end
        end else begin
            // Look up the BTB with the current PC
            if (btb_pc[index] == pc_in) begin
                // BTB hit: we found a matching PC
                hit <= 1;
                predicted_target <= btb_target[index];
            end else begin
                // BTB miss
                hit <= 0;
                predicted_target <= 0;
            end
            
            // If the branch is taken, update the BTB
            if (branch_taken) begin
                btb_pc[index] <= pc_in;
                btb_target[index] <= target_addr;
            end
        end
    end
    
endmodule


module compare  


// All comparisons are signed by default; unsigned comparisons can be implemented
// by treating operands as unsigned integers or through a future \texttt{CMPU} 
//variant if required.  
// The \texttt{EV}/\texttt{OD} conditions allow efficient branching on bit~0, 
//which is useful for address alignment tests, distinguishing even/odd indices, 
// and loop unrolling.

// Because all comparison and branch instructions share this common encoding and
// logic, the branch unit requires only a single multiplexer and no dedicated 
// condition register, simplifying both the datapath and verification.


wire Z  = (result == 64'd0);  // Zero flag
wire S  = result[63];         // Sign flag (for signed comparisons)
wire B0 = result[0];          // Least significant bit (for EV/OD)

wire cond_EQ =  Z;
wire cond_LT =  S;
wire cond_GT = (~S) & (~Z);
wire cond_EV = ~B0;

wire cond_NE = ~Z;
wire cond_GE = ~S;
wire cond_LE =  S | Z;
wire cond_OD =  B0;

assign branch_taken =
    (cond == 3'b000) ? cond_EQ :
    (cond == 3'b001) ? cond_LT :
    (cond == 3'b010) ? cond_GT :
    (cond == 3'b011) ? cond_EV :
    (cond == 3'b100) ? cond_NE :
    (cond == 3'b101) ? cond_GE :
    (cond == 3'b110) ? cond_LE :
                       cond_OD;

endmodule
