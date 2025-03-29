// Copyright 2022 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Bowen Wang

#ifndef _ALLOC_PARTITION_H_
#define _ALLOC_PARTITION_H_
// ------ Dynamic Data Pointers ------ //
#define NUM_TILES (128)
// void* volatile Region_A[NUM_TILES] __attribute__((section(".l1")));
// void* volatile Region_B[NUM_TILES] __attribute__((section(".l1")));
// void* volatile Region_C[NUM_TILES] __attribute__((section(".l1")));
// void* volatile Region_D[NUM_TILES] __attribute__((section(".l1")));

// ------ Partition Status Info ------ //
#define NUM_ELEMENTS_PER_ROW (4096)
#define NUM_PART_REGION      (4) 

typedef struct {
  float *data_addr;     // trace which matrix belong to this partition
  uint32_t status;        // set to 1 if used
} partition_status_t;

// partition_status_t volatile partition_status[NUM_PART_REGION] __attribute__((section(".l1")));


void alloc_matrix(float *volatile * target, uint32_t size, uint32_t group_factor, uint32_t num_matrix);

void free_matrix(float *__restrict__ heap_matrix, uint32_t part_id, uint32_t core_id);

void free_alloc(uint32_t core_id);

#endif