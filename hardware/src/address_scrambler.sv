// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51

// Description: Scrambles the address in such a way, that part of the memory is accessed
// sequentially and part is interleaved.

// Author: Samuel Riedel <sriedel@iis.ee.ethz.ch>
// Last editor: Elio Wanner <ewanner@ethz.ch>

module address_scrambler #(
  parameter int unsigned AddrWidth             = 32,        // Max length of address (including unused bits)
  parameter int unsigned ByteOffset            = 2,         // [bits] Selecting bits within one bank's row
  parameter int unsigned NumTiles              = 2,         // Number of tiles
  parameter int unsigned NumBanksPerTile       = 2,         // Number of banks per tile = Bank offset
  parameter bit          Bypass                = 0,         // To skip scrambling
  parameter int unsigned SeqMemSizePerTile     = 4*1024,    // [bytes] Size of stack within one tile - Defining the stack region
  parameter int unsigned MemSizePerTile        = 8*4*1024,  // [bytes] Size of stack+heap within one tile (all the memory in one tile)
  parameter int unsigned MemSizePerRow         = 4*4*1024   // [bytes] Size of one row spanning all tiles (one full row without scrambling)
) (
  input  logic [AddrWidth-1:0]      address_i,              // Input address
  output logic [AddrWidth-1:0]      address_o,              // Output address

  // Heap Settings (max 4 partitions and 128 rows per partition)
  input  logic [3:0][7:0]           group_factor_i,         // Number of tiles per partition (1,2,4,8,16,32,64,128)
  input  logic [3:0][7:0]           allocated_size_i,       // Number of rows assigned to each partition (max 128)
  input  logic [3:0][AddrWidth-1:0] start_addr_scheme_i     // Start address of each partition
);

  // Stack Sequential Settings
  localparam int unsigned BankOffsetBits    = $clog2(NumBanksPerTile);          // Number of bits needed for addressing a bank within a tile
  localparam int unsigned TileIdBits        = $clog2(NumTiles);                 // Number of bits needed for addressing a tile
  localparam int unsigned SeqPerTileBits    = $clog2(SeqMemSizePerTile);        // Number of bits needed for addressing stack within one tile
  localparam int unsigned SeqTotalBits      = SeqPerTileBits + TileIdBits;      // Number of bits needed for addressing stack within all tiles
  localparam int unsigned ConstantBitsLSB   = BankOffsetBits + ByteOffset;      // Number of bits needed for addressing bank+within bank 
  localparam int unsigned ScrambleBits      = SeqPerTileBits - ConstantBitsLSB; // Number of bits needed for addressing row within stack

  // Heap Sequential Settings
  localparam int unsigned HeapSeqPerTileBits = $clog2(MemSizePerTile);               // Number of bits needed for addressing stack+heap within one tile
  localparam int unsigned HeapSeqTotalBits   = HeapSeqPerTileBits + TileIdBits;      // Number of bits needed for addressing stack+heap within all tiles
  localparam int unsigned RowIndexBits       = HeapSeqPerTileBits - ConstantBitsLSB; // Number of bits needed for addressing row within stack+heap within one tile
  

  if (Bypass || NumTiles < 2) begin
    assign address_o = address_i;            // No Scrambling

  end else begin
    // ---------------------------------------------------------------------------
    // Stack Region signals
    // ---------------------------------------------------------------------------
    // We wrap around at the end of the tile.
    logic [ScrambleBits-1:0]    scramble;    // Address bits that contain row within stack
    logic [TileIdBits-1:0]      tile_id;     // Address bits that contain tile

    // Get content from scramble and tile_id parts of the address
    assign scramble = address_i[SeqPerTileBits-1:ConstantBitsLSB];
    assign tile_id  = address_i[SeqTotalBits-1:SeqPerTileBits];

    // ---------------------------------------------------------------------------
    // Heap Region signals
    // ---------------------------------------------------------------------------
    // We wrap around at the end of the partition. 
    // We need to shift by the number of bits that are needed to address the partition size
    logic [3:0][2:0] shift_index;     // Number of bits needed to shift for given partition size (horizontal, tiles)
    logic [3:0][2:0] shift_index_sc;  // Number of bits needed to shift for given partition size (vertical, rows)
    
    for (genvar i = 0; i < 4; i++) begin : gen_shift_index
        // This is a lookup table taking log2
        always_comb begin
            // Nr of tiles per partition
            case(group_factor_i[i])
                128: shift_index[i] = 7;
                64:  shift_index[i] = 6;
                32:  shift_index[i] = 5;
                16:  shift_index[i] = 4;
                8:   shift_index[i] = 3;
                4:   shift_index[i] = 2;
                2:   shift_index[i] = 1;
                default: shift_index[i] = 0;
            endcase
            // Nr of rows per partition
            case(allocated_size_i[i])
                128: shift_index_sc[i] = 7;
                64:  shift_index_sc[i] = 6;
                32:  shift_index_sc[i] = 5;
                16:  shift_index_sc[i] = 4;
                8:   shift_index_sc[i] = 3;
                4:   shift_index_sc[i] = 2;
                2:   shift_index_sc[i] = 1;
                default: shift_index_sc[i] = 0;
            endcase
        end
    end

    logic [RowIndexBits-1:0]      post_scramble_row_index;
    logic [TileIdBits-1:0]        post_scramble_tile_id;

    logic [3:0][RowIndexBits-1:0] mask_row_index, mask_row_index_n;
    logic [3:0][TileIdBits-1:0]   mask_tile_id,   mask_tile_id_n;

    logic [TileIdBits-1:0]        heap_tile_id;

    // ---------------------------------------------------------------------------
    // Generate the masks for the heap regions.
    // ---------------------------------------------------------------------------
    for (genvar j = 0; j < 4; j++) begin : gen_mask
      assign mask_row_index[j] = (shift_index_sc[j] == 0) 
                                ? {RowIndexBits{1'b0}} 
                                : ({RowIndexBits{1'b1}} >> (RowIndexBits - shift_index_sc[j]));

      assign mask_tile_id[j]   = (shift_index[j] == 0)    
                                ? {TileIdBits{1'b0}}   
                                : ({TileIdBits{1'b1}}   >> (TileIdBits   - shift_index[j]));
      
      assign mask_row_index_n[j] = ~mask_row_index[j];
      assign mask_tile_id_n[j]   = ~mask_tile_id[j];
    end

    assign heap_tile_id = address_i[(TileIdBits+ConstantBitsLSB-1):ConstantBitsLSB];

    // ---------------------------------------------------------------------------
    // Macro to handle the "heap region" scrambling logic.
    // idx is the index to define the partition.
    // ---------------------------------------------------------------------------
    `define SCRAMBLE_HEAP_REGION(idx)                                                   \
      post_scramble_row_index |= (address_i >> (ConstantBitsLSB + shift_index[idx]))    \
                                & mask_row_index[idx];                                  \
      post_scramble_row_index |= (address_i >> (ConstantBitsLSB + TileIdBits))          \
                                & mask_row_index_n[idx];                                \
                                                                                        \
      post_scramble_tile_id   |= heap_tile_id & mask_tile_id[idx];                      \
      post_scramble_tile_id   |= (address_i >> (ConstantBitsLSB + shift_index_sc[idx])) \
                                & mask_tile_id_n[idx];                                  \
                                                                                        \
      address_o[HeapSeqTotalBits-1 : ConstantBitsLSB] = { post_scramble_row_index,      \
                                                        post_scramble_tile_id }

    // ---------------------------------------------------------------------------
    // Main logic of the address scrambler.
    // ---------------------------------------------------------------------------
    always_comb begin
      // $display("[scrambler debug] address_i: %h -> address_o: %h", address_i, address_o);
      // $display("[scrambler debug] start_addr_scheme_i[0]: %h", start_addr_scheme_i[0]);
      // $display("[scrambler debug] start_addr_scheme_i[1]: %h", start_addr_scheme_i[1]);
      // $display("[scrambler debug] start_addr_scheme_i[2]: %h", start_addr_scheme_i[2]);
      // $display("[scrambler debug] start_addr_scheme_i[3]: %h", start_addr_scheme_i[3]);


      // Default: unscrambled
      address_o                 = address_i;
      post_scramble_row_index   = 'b0;
      post_scramble_tile_id     = 'b0;

      // Stack Region (wrap at end of tile)
      if (address_i < (NumTiles * SeqMemSizePerTile)) begin
        address_o[SeqTotalBits-1 : ConstantBitsLSB] = { scramble, tile_id };
        //$display("[scrambler debug] IN STACK address_i: %h -> address_o: %h", address_i, address_o);

      // 4 Heap Regions (wrap at end of partition)
      end else if ( (address_i >= start_addr_scheme_i[0]) &&
                    (address_i <  start_addr_scheme_i[0] + MemSizePerRow * allocated_size_i[0]) ) begin
        `SCRAMBLE_HEAP_REGION(0);
        // $display("[scrambler debug] IN REGION 0 address_i: %h -> address_o: %h", address_i, address_o);
        // $display("[scrambler debug] allocated_size_i[0]: %h", allocated_size_i[0]);
        // $display("[scrambler debug] group_factor_i[0]: %h", group_factor_i[0]);
        // $display("[scrambler debug] shift_index[0]: %h", shift_index[0]);
        // $display("[scrambler debug] shift_index_sc[0]: %h", shift_index_sc[0]);
        // display uper and lower bounds
        // $display("[scrambler debug] LOWER: %h", start_addr_scheme_i[0]);
        // $display("[scrambler debug] UPPER: %h", start_addr_scheme_i[0] + MemSizePerRow * allocated_size_i[0]);

      end else if ( (address_i >= start_addr_scheme_i[1]) &&
                    (address_i <  start_addr_scheme_i[1] + MemSizePerRow * allocated_size_i[1]) ) begin
        `SCRAMBLE_HEAP_REGION(1);
        // $display("[scrambler debug] IN REGION 1 address_i: %h -> address_o: %h", address_i, address_o);

      end else if ( (address_i >= start_addr_scheme_i[2]) &&
                    (address_i <  start_addr_scheme_i[2] + MemSizePerRow * allocated_size_i[2]) ) begin
        `SCRAMBLE_HEAP_REGION(2);

      end else if ( (address_i >= start_addr_scheme_i[3]) &&
                    (address_i <  start_addr_scheme_i[3] + MemSizePerRow * allocated_size_i[3]) ) begin
        `SCRAMBLE_HEAP_REGION(3);
      end
    end


  end

  // Check for unsupported configurations
  if (NumBanksPerTile < 2)
    $fatal(1, "NumBanksPerTile must be greater than 2. The special case '1' is currently not supported!");
  if (SeqMemSizePerTile % (2**ByteOffset*NumBanksPerTile) != 0)
    $fatal(1, "SeqMemSizePerTile must be a multiple of BankWidth*NumBanksPerTile!");
endmodule : address_scrambler
