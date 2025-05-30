// Copyright 2021 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This Kernel implements matrix multiplication using address scrambling.

// Author: Domenic Wüthrich, ETH Zurich
//         Elio Wanner, ETH Zurich

// This kernel runs but does currently doesn't have performance improvements through address scrambling.

// To run (if config not changed the clean can be removed): 
// MinPool:  make -C spatz_apps/auto_benchmark clean fmatmul-flex size=16  cores=4   config=minpool_spatz4_fpu  sim=sim log=false
//           make -C spatz_apps/auto_benchmark clean fmatmul-flex size=64  cores=4   config=minpool_spatz4_fpu  sim=sim log=false
// MemPool:  make -C spatz_apps/auto_benchmark clean fmatmul-flex size=64  cores=64  config=mempool_spatz4_fpu  sim=sim log=false
//           make -C spatz_apps/auto_benchmark clean fmatmul-flex size=256 cores=64  config=mempool_spatz4_fpu  sim=sim log=false
// TeraPool: make -C spatz_apps/auto_benchmark clean fmatmul-flex size=128 cores=128 config=terapool_spatz8_fpu sim=sim log=false
//           make -C spatz_apps/auto_benchmark clean fmatmul-flex size=256 cores=128 config=terapool_spatz8_fpu sim=sim log=false

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "data/data_gemm.h"
#include "kernel/sp-fmatmul.c"
#include "printf.h"
#include "alloc.h"
#include "runtime.h"
#include "synchronization.h"
#include "encoding.h"

#include "alloc_partition.h"

partition_status_t volatile partition_status[NUM_PART_REGION] __attribute__((section(".l1")));

#include "dma.h"

// Initialize the matrices
void init_matrix(float *matrix, const float *src,
                 const uint32_t rows_start, const uint32_t rows_end,
                 const uint32_t num_columns) {
  for (uint32_t i = rows_start; i < rows_end; ++i) {
    for (uint32_t j = 0; j < num_columns; ++j) {
      matrix[i * num_columns + j] = src[i * num_columns + j];
    }
  }
}

// Verify the matrices
int verify_matrix(float *matrix, const float *checksum,
                  const uint32_t num_rows, const uint32_t num_columns) {
  for (uint32_t i = 0; i < num_rows; ++i) {
    float sum = 0;
    for (uint32_t j = 0; j < num_columns; ++j) {
      sum += (float)matrix[i * num_columns + j];
    }

    float diff = (float)sum - (float)checksum[i];//(float)matrix[i * num_columns + j] - (float)checksum[i * num_columns + j];
    float prec = (float)0.001;
    if (diff < 0)
      diff = -diff;
    if (diff > prec) {
      return (i) == 0 ? -1 : (int)(i);
    }
  // }
  }
  return 0;
}

void print_matrix(float const *matrix, uint32_t num_rows,
                  uint32_t num_columns) {
  printf("0x%8X\n", (uint32_t)matrix);
  for (uint32_t i = 0; i < num_rows; ++i) {
    for (uint32_t j = 0; j < num_columns; ++j) {
      printf("%5u ", (uint32_t)matrix[i * num_columns + j]);
    }
    printf("\n");
  }
}

void print_float_matrix(float const *matrix, uint32_t num_rows,
  uint32_t num_columns) {
  printf("0x%8X\n", (uint32_t)matrix);
  for (uint32_t i = 0; i < num_rows; ++i) {
  for (uint32_t j = 0; j < num_columns; ++j) {
  float val = matrix[i * num_columns + j];
  int int_part = (int)val;
  int decimal_part = (int)((val - int_part) * 10);
  if (decimal_part < 0) decimal_part = -decimal_part;
  printf("%4d.%1d ", int_part, decimal_part);
  }
  printf("\n");
  }
}

// Matrix A: MxN
// Matrix B: NxP
// Matrix C: MxP

int main() {
  const uint32_t num_cores = mempool_get_core_count();
  const uint32_t cores_per_group = num_cores / NUM_GROUPS;
  const uint32_t cid = mempool_get_core_id();
  const uint32_t core_gid = cid % cores_per_group;
  const uint32_t gid = cid / cores_per_group;

  const uint32_t active_groups = 4;
  const uint32_t active_cores = cores_per_group * active_groups;
  const uint32_t is_core_active = cid < active_cores;

  uint32_t timer_start, timer_end, timer;

  uint32_t m_start, m_end;
  uint32_t p_start, p_end;
  uint32_t kernel_size;

  // Initialize multicore barrier
  mempool_barrier_init(cid);

  // Reset timer
  timer = (uint32_t)-1;

  // Set matrix dimension (Diyou: this can be left unchanged)
  kernel_size = 4; // STILL HARDCODED

  // Block dimension of group
  const uint32_t dim_group = gemm_l.M / active_groups;
  // Number of parallel cores in m direction
  const uint32_t split_m_count = dim_group / kernel_size;

  if (split_m_count < cores_per_group) {
    // Split P dimension up
    const uint32_t split_p_count = cores_per_group / split_m_count;
    p_start = gemm_l.P / split_p_count * (core_gid % split_p_count);
    p_end   = gemm_l.P / split_p_count * ((core_gid % split_p_count) + 1);
    m_start = dim_group * gid + kernel_size * (core_gid / split_p_count);
    m_end   = dim_group * gid + kernel_size * (core_gid / split_p_count + 1);
  } else {
    // Work over complete P dimension
    p_start = 0;
    p_end   = gemm_l.P;
    m_start = dim_group * gid + (dim_group / cores_per_group) * core_gid;
    m_end   = dim_group * gid + (dim_group / cores_per_group) * (core_gid + 1);
  }

  if (cid == 0) {
    printf("Core %d: m_start=%d, m_end=%d, p_start=%d, p_end=%d\n", cid, m_start, m_end, p_start, p_end);
    printf("Kernel size: %d\n", kernel_size);
  }

  // Wait for all cores to finish
  mempool_barrier(num_cores);

  // Code for dynamic memory allocation -----------------------------------------------------
  extern uint32_t __heap_start;     // Start of interleaved heap
  extern uint32_t __heap_seq_start; // Start of sequential heap, MinPool=0x4000
  extern uint32_t __l1_end;         // End of total L1 memory (and sequential heap)
  extern alloc_t alloc_l1;

  bool use_sequential_region = true;

  // Sequential heap parameters
  uint32_t group_factor = num_cores; // MinPool=4, MemPool=64, TeraPool=128
  uint32_t num_partition = mempool_get_tile_count() / group_factor;

  uint32_t tiles_per_partition = 1;

  if (cid == 0) {
    printf("In sp-fmatmul-flex script\n");
    // Initialize the allocator
    alloc_init(&alloc_l1, (void*)&__heap_start, (uint32_t)&__l1_end - (uint32_t)&__heap_start);
    // Initialize and reset the sequetial heap
    mempool_dynamic_heap_alloc_init(cid, group_factor);
    mempool_dynamic_heap_alloc_reset(cid, group_factor, (uint32_t)&__heap_seq_start);
  }

  // init partition info
  if (cid == 0){
    for (uint32_t i=0; i<NUM_PART_REGION; ++i){
        partition_status[i].status    = 0;
        partition_status[i].data_addr = NULL;
    }
  }

  // Define input matrices globally
  static float *volatile *a = NULL;
  static float *volatile *b = NULL;
  static float *volatile *c = NULL;

  const uint32_t dim_a = gemm_l.M * gemm_l.N;
  const uint32_t dim_b = gemm_l.N * gemm_l.P;
  const uint32_t dim_c = gemm_l.M * gemm_l.P;

  // Initialize multicore barrier
  mempool_barrier_init(cid);

  if (cid == 0) {
    if (use_sequential_region){
      // Dynamic memory allocation (sequential region)-----------------------
      printf("Using sequential region for A\n");
      alloc_matrix(&a, dim_a, 1, num_partition);            // A in sequential region, with folding after each tile
      //alloc_matrix(&c, dim_c, 1, num_partition); 
      b = (float *)simple_malloc(dim_b * sizeof(float));    // B in interleaved region
      c = (float *)simple_malloc(dim_c * sizeof(float));    // C in interleaved region
    } else{
      // Dynamic memory allocation (interleaved region)----------------------
      a = (float *)simple_malloc(dim_a * sizeof(float));
      b = (float *)simple_malloc(dim_b * sizeof(float));
      c = (float *)simple_malloc(dim_c * sizeof(float));
    }

    printf("finished allocation\n");
    
    // Once DMA is fixed by Bowen, these lines should be used for DMA transfer
    // dma_memcpy_ModeSel(a, gemm_A_dram, (gemm_l.M * gemm_l.N) * sizeof(float), DMA_FAST);
    // dma_memcpy_blocking(b, gemm_B_dram, (gemm_l.N * gemm_l.P) * sizeof(float));
    // dma_memcpy_ModeSel(c, gemm_C_dram, (gemm_l.M * gemm_l.P) * sizeof(float), DMA_FAST);

    dma_memcpy_blocking(a, gemm_A_dram, dim_a * sizeof(float));
    dma_memcpy_blocking(b, gemm_B_dram, dim_b * sizeof(float));
    dma_memcpy_blocking(c, gemm_C_dram, dim_c * sizeof(float));

    init_matrix(r, gemm_checksum, 0, 1, gemm_l.M);
    printf("finished copying\n");
  }

  // Wait for all cores to finish
  mempool_barrier(num_cores);

  // Calculate matmul
  if (is_core_active) {
    // Start dump
    if (cid == 0){
      mempool_start_benchmark();
      //timer_start = mempool_get_timer();
    }

    if (kernel_size == 2) {
      timer_start = mempool_get_timer();
      matmul_2xVL(c, a, b, m_start, m_end, gemm_l.N, gemm_l.P, p_start, p_end);
    } else if (kernel_size == 4) {
      timer_start = mempool_get_timer();
      matmul_4xVL(c, a, b, m_start, m_end, gemm_l.N, gemm_l.P, p_start, p_end);
    } else if (kernel_size == 8) {
      timer_start = mempool_get_timer();
      matmul_8xVL(c, a, b, m_start, m_end, gemm_l.N, gemm_l.P, p_start, p_end);
    } else {
      return -2;
    }
  }

  // Wait for all cores to finish matmul
  mempool_barrier(num_cores);

  if (cid == 0) {
    timer_end = mempool_get_timer();
    timer = (timer_end - timer_start);
    mempool_stop_benchmark();
  }
    

  // Check and display results
  if (cid == 0) {
    long unsigned int performance = 1000 * 2 * gemm_l.M * gemm_l.P * gemm_l.N / timer;
    long unsigned int utilization = performance / (2 * active_cores * N_FPU);

    printf("\n----- (%dx%d) sp fmatmul -----\n", gemm_l.M, gemm_l.P);
    printf("The execution took %u cycles.\n", timer);
    printf("The performance is %u OP/1000cycle (%u%%o utilization).\n",
           performance, utilization);
  }

  if (cid == 0) {
    int error = verify_matrix((float *)c, (const float *)r, gemm_l.M, gemm_l.P);
    // int error = verify_matrix((float *)c, (const float *)gemm_checksum, gemm_l.M, gemm_l.P);

    if (error != 0) {
      printf("Error core %d: c[%d]=%x\n", cid, error, (int)c[error]);
      //return error;
    } else {
      printf("Checksum is correct.\n");
    }
  }

  // Wait for core 0 to finish displaying results
  mempool_barrier(num_cores);

  return 0;
}
