// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/* This file will get processed by the precompiler to expand all macros. */

MEMORY {
  l1 (R) : ORIGIN = 0x00000000, LENGTH = (NUM_CORES * N_FU * BANKING_FACTOR * L1_BANK_SIZE)
  l2     : ORIGIN = L2_BASE   , LENGTH = L2_SIZE
  rom (R): ORIGIN = BOOT_ADDR , LENGTH = 0x00001000
}

SECTIONS {
  // Start end end of memories
  __l1_start = ORIGIN(l1);
  __l1_end = ORIGIN(l1) + LENGTH(l1);
  __l2_start = ORIGIN(l2);
  __l2_end = ORIGIN(l2) + LENGTH(l2);
  __rom_start = ORIGIN(rom);
  __rom_end = ORIGIN(rom) + LENGTH(rom);

  // Stack size
  __stack_start = __l1_start;
  __stack_end = __l1_start + (NUM_CORES * N_FU * STACK_SIZE);

  // Sequential region size
  __seq_start = __l1_start;
  __seq_end = __l1_start + (NUM_CORES * N_FU * SEQ_MEM_SIZE);

  // Heap size (start address is re-assigned in link.ld)
  __heap_start = __l1_end;
  __heap_seq_start = __l1_start + (NUM_CORES * N_FU * 2 * L1_BANK_SIZE);
  __heap_end = __l1_end;

  // Hardware register location
  eoc_reg                = 0x40000000;
  wake_up_reg            = 0x40000004;
  wake_up_group_reg      = 0x40000008;
  tcdm_start_address_reg = 0x4000000C;
  tcdm_end_address_reg   = 0x40000010;
  nr_cores_address_reg   = 0x40000014;
  ro_cache_enable        = 0x40000018;
  ro_cache_flush         = 0x4000001C;
  ro_cache_start_0       = 0x40000020;
  ro_cache_end_0         = 0x40000024;
  ro_cache_start_1       = 0x40000028;
  ro_cache_end_1         = 0x4000002C;
  ro_cache_start_2       = 0x40000030;
  ro_cache_end_2         = 0x40000034;
  ro_cache_start_3       = 0x40000038;
  ro_cache_end_3         = 0x4000003C;

  wake_up_tile_g0_reg = 0x40000040;
  wake_up_tile_g1_reg = 0x40000044;
  wake_up_tile_g2_reg = 0x40000048;
  wake_up_tile_g3_reg = 0x4000004C;
  wake_up_tile_g4_reg = 0x40000050;
  wake_up_tile_g5_reg = 0x40000054;
  wake_up_tile_g6_reg = 0x40000058;
  wake_up_tile_g7_reg = 0x4000005C;

  partition_reg       = 0x40000060;

  start_addr_scheme0_reg = 0x40000064;
  start_addr_scheme1_reg = 0x40000068;
  start_addr_scheme2_reg = 0x4000006C;
  start_addr_scheme3_reg = 0x40000070;

  partition1_reg       = 0x40000074;
  partition2_reg       = 0x40000078;
  partition3_reg       = 0x4000007C;

  allocated_size0_reg = 0x40000080;
  allocated_size1_reg = 0x40000084;
  allocated_size2_reg = 0x40000088;
  allocated_size3_reg = 0x4000008C;

  dma_mode_reg        = 0x40000090;

  fake_uart              = 0xC0000000;
}
