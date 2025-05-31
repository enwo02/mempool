// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Bowen Wang
// Editor: Elio Wanner

#ifndef _ALLOC_PARTITION_H_
#define _ALLOC_PARTITION_H_
// ------ Dynamic Data Pointers ------ //
#define NUM_TILES (NUM_CORES / NUM_CORES_PER_TILE) // Minpool: 4 tiles

// ------ Partition Status Info ------ //
#define NUM_ELEMENTS_PER_ROW (BANKING_FACTOR * N_FPU * NUM_TILES) // Minpool: 16 * 4 = numberOfBanksPerTile * numberOfTiles 
#define NUM_PART_REGION      (4)  // STILL HARDCODED, for dotp-dyn this is 2
// extern uint32_t NUM_PART_REGION; // number of partition regions

typedef struct {
  float *data_addr;       // trace which matrix belong to this partition
  uint32_t status;        // set to 1 if used
} partition_status_t;

// Comment for dotp-dyn and fmatmul-flex
//partition_status_t volatile partition_status[NUM_PART_REGION] __attribute__((section(".l1")));


void alloc_matrix(float *volatile * target, uint32_t size, uint32_t group_factor, uint32_t num_matrix);

void free_matrix(float *__restrict__ heap_matrix, uint32_t part_id, uint32_t core_id);

void free_alloc(uint32_t core_id);

#endif