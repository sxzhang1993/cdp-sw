#include "mat.h"
#include <chrono>
#include <fstream>
#include <ios>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <stdlib.h>

#define M 3      // match
#define MM -3    // mismatch
#define W -2     // gap score
#define A_LEN 4096 
#define B_LEN 4096 
#define max(a, b) (((a) > (b)) ? (a) : (b)) // return maximum of two values
#define min(a, b) (((a) < (b)) ? (a) : (b)) // return minimum of two values

// Forward declarations of scoring kernel
__global__ void fill_gpu(Matrixsz h, Matrixsz d, char seqA[], char seqB[],
                         const int *k);

// generate random sequence of length n
void seq_gen(int n, char seq[]) {
  for (int i = 0; i < n; i++) {
    int base = rand() % 4;
    switch (base) {
    case 0:
      seq[i] = 'A';
      break;
    case 1:
      seq[i] = 'T';
      break;
    case 2:
      seq[i] = 'C';
      break;
    case 3:
      seq[i] = 'G';
      break;
    }
  }
}

int fill_cpu(Matrixsz h, Matrixsz d, char seqA[], char seqB[]) {

  int full_max_id = 0;
  int full_max_val = 0;

  for (int i = 1; i < h.height; i++) {
    for (int j = 1; j < h.width; j++) {

      // scores
      int max_score = 0;
      int direction = 0;
      int tmp_score;
      int sim_score;

      // comparison positions
      int id = i * h.width + j;                  // current cell
      int abov_id = (i - 1) * h.width + j;       // above cell, 1
      int left_id = i * h.width + (j - 1);       // left cell, 2
      int diag_id = (i - 1) * h.width + (j - 1); // upper-left diagonal cell, 3

      // above cell
      tmp_score = h.elements[abov_id] + W;
      if (tmp_score > max_score) {
        max_score = tmp_score;
        direction = 1;
      }

      // left cell
      tmp_score = h.elements[left_id] + W;
      if (tmp_score > max_score) {
        max_score = tmp_score;
        direction = 2;
      }

      // diagonal cell (preferred)
      char baseA = seqA[j - 1];
      char baseB = seqB[i - 1];
      if (baseA == baseB) {
        sim_score = M;
      } else {
        sim_score = MM;
      }

      tmp_score = h.elements[diag_id] + sim_score;
      if (tmp_score >= max_score) {
        max_score = tmp_score;
        direction = 3;
      }

      // assign scores and direction
      h.elements[id] = max_score;
      d.elements[id] = direction;

      if (max_score > full_max_val) {
        full_max_id = id;
        full_max_val = max_score;
      }
    }
  }

  std::cout << "Max score of " << full_max_val;
  std::cout << " at id: " << full_max_id << std::endl;
  return full_max_id;
}

__global__ void fill_gpu(float h[4097*4097], float d[4097*4097], char seqA[], char seqB[],
                         const int k, int max_id_val[]) {

  // scores
  int max_score = 0;
  int direction = 0;
  int tmp_score;
  int sim_score;

  // row and column index depending on anti-diagonal
  int i = threadIdx.x + 1 + blockDim.x * blockIdx.x;
  if (k > A_LEN + 1) {
    i += (k - A_LEN);
  }
  int j = ((k) - i) + 1;
  int id = i * 4097 + j; //SZZ value 
  // printf("round: %d\n", k);
  // printf("threadIdx.x: %d, blockDim.x: %d, blockIdx.x: %d\n i: %d\n", threadIdx.x, blockDim.x, blockIdx.x, k);
  // printf("threadIdx.x: %d, blockDim.x: %d, blockIdx.x: %d\n i: %d, j: %d\n id: %d\n", threadIdx.x, blockDim.x, blockIdx.x, i, j, id);
  // printf("width: %d", h.width);
  // printf("height: %d", h.height);

  // comparison positions
  // int id = i * 4097 + j; //SZZ value
  int abov_id = (i - 1) * 4097 + j;       // above cell, 1
  int left_id = i * 4097 + (j - 1);       // left cell, 2
  int diag_id = (i - 1) * 4097 + (j - 1); // upper-left diagonal cell, 3

  // above cell
//  tmp_score = 4097*4097[abov_id] + W;
  //SZZ : tempoary diabled
  if (tmp_score > max_score) {
    max_score = tmp_score;
    direction = 1;
  }

  // left cell
 // tmp_score = 4097*4097[left_id] + W;
  if (tmp_score > max_score) {
    max_score = tmp_score;
    direction = 2;
  }

  // similarity score for diagonal cell
  char baseA = seqA[j - 1];
  char baseB = seqB[i - 1];
  if (baseA == baseB) {
    sim_score = M;
  } else {
    sim_score = MM;
  }

  // diagonal cell (preferred)
 // tmp_score = 4097*4097[diag_id] + sim_score;
  if (tmp_score >= max_score) {
    max_score = tmp_score;
    direction = 3;
  }

  // assign scores and direction
  // printf("id_check: %d\n", id);
 // h.elements[id] = max_score;
 // d.elements[id] = direction;
//SZZ: tempoary diableed
  // save max score and position
  if (max_score > max_id_val[1]) {
    max_id_val[0] = id;
    max_id_val[1] = max_score;
  }
}

// traceback: starting at the highest score and ending at a 0 score
void traceback(Matrixsz d, int max_id, char seqA[], char seqB[],
               std::vector<char> &seqA_aligned,
               std::vector<char> &seqB_aligned) {

  int max_i = max_id / d.width;
  int max_j = max_id % d.width;

  // traceback algorithm from maximum score to 0
  while (max_i > 0 && max_j > 0) {

    int id = max_i * d.width + max_j;
    int dir = d.elements[id];

    switch (dir) {
    case 1:
      --max_i;
      seqA_aligned.push_back('-');
      seqB_aligned.push_back(seqB[max_i]);
      break;
    case 2:
      --max_j;
      seqA_aligned.push_back(seqA[max_j]);
      seqB_aligned.push_back('-');
      break;
    case 3:
      --max_i;
      --max_j;
      seqA_aligned.push_back(seqA[max_j]);
      seqB_aligned.push_back(seqB[max_i]);
      break;
    case 0:
      max_i = -1;
      max_j = -1;
      break;
    }
  }
}

// print aligned sequnces
void io_seq(std::vector<char> &seqA_aligned, std::vector<char> &seqB_aligned) {

  std::cout << "Aligned sub-sequences of A and B: " << std::endl;
  int align_len = seqA_aligned.size();
  std::cout << "   ";
  for (int i = 0; i < align_len + 1; ++i) {
    std::cout << seqA_aligned[align_len - i];
  }
  std::cout << std::endl;

  std::cout << "   ";
  for (int i = 0; i < align_len + 1; ++i) {
    std::cout << seqB_aligned[align_len - i];
  }
  std::cout << std::endl << std::endl;
}

// input output function to visualize matrix
void io_score(std::string file, Matrixsz h, char seqA[], char seqB[]) {
  std::ofstream myfile_tsN;
  myfile_tsN.open(file);

  // print seqA
  myfile_tsN << '\t' << '\t';
  for (int i = 0; i < A_LEN; i++)
    myfile_tsN << seqA[i] << '\t';
  myfile_tsN << std::endl;

  // print vertical seqB on left of matrix
  for (int i = 0; i < h.height; i++) {
    if (i == 0) {
      myfile_tsN << '\t';
    } else {
      myfile_tsN << seqB[i - 1] << '\t';
    }
    for (int j = 0; j < h.width; j++) {
      myfile_tsN << h.elements[i * h.width + j] << '\t';
    }
    myfile_tsN << std::endl;
  }
  myfile_tsN.close();
}

__global__ void cdprun(int const iSize, int iDepth, float h[(A_LEN+1)*(B_LEN+1)], float d[(A_LEN+1)*(B_LEN+1)], char seqA[], char seqB[])

{
 // Matrixsz d_h(A_LEN + 1, B_LEN + 1,1);
 // Matrixsz d_d(A_LEN + 1, B_LEN + 1,1);
	//SZZ still cannot use this class -- confirmed cannot use ''class'' in __global__
//float d_h[A_LEN+1][B_LEN+1];
	
//float d_h[A_LEN+1][B_LEN+1];
  // for (int zh = 0; zh < A_LEN+1; zh++){
   // for (int zw = 0; zw < B_LEN+1; zw++)
    //    d_h[zh][zw] = h[zh][zw];
  // }
//float d_d[A_LEN+1][B_LEN+1];
// for (int sh = 0; sh < A_LEN+1; sh++){
   // for (int sw = 0; sw < B_LEN+1; sw++)
  //      d_d[sh][sw] = d[sh][sw];
//}
//SZZ: 2-d matrix [][] cannot use to fill_gpu?
__shared__ float d_h[(A_LEN+1)*(B_LEN+1)];
	for (int zh = 0; zh < 4097; zh++){
    for (int zw = 0; zw < 4097; zw++)
        d_h[(A_LEN+1) * zh + zw] = h[(A_LEN+1) * zh + zw];
}

__shared__ float d_d[(A_LEN+1)*(B_LEN+1)];
        for (int sh = 0; sh < 4097; sh++){
    for (int sw = 0; sw < 4097; sw++)
        d_h[(A_LEN+1) * sh + sw] = h[(A_LEN+1) * sh + sw];
}

//SZZ: error:            argument types are: (float [16785409], float [16785409], char *, char *, int, int *)

//SZZ: 1-d matrix cannot work


// SZZ:  error: initialization with "{...}" expected for aggregate objecti --solved

// float d_h[][];
// float d_d[][];

 //SZZ error message: align.cu(262): error: an array may not have elements of this type -- solved

// SZZ: error: no instance of overloaded function "fill_gpu" matches the argument list
          //  argument types are: (float [4097][4097], float [4097][4097], char *, char *, int, int *)

  // std::cout << "CDP GPU result: " << std::endl;

  // allocate and transfer sequence data to device
  char *d_seqA, *d_seqB;

  //SZZ cant use cuda... function in __global__
  //cudaMalloc(&d_seqA, A_LEN * sizeof(char));
  //cudaMalloc(&d_seqB, B_LEN * sizeof(char));
  //cudaMemcpy(d_seqA, seqA, A_LEN * sizeof(char), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_seqB, seqB, B_LEN * sizeof(char), cudaMemcpyHostToDevice);
        //SZZ cannot use Matrix:: in global
      //  float d_h;
 // std::cout << "ALEN: "  << A_LEN << std::endl;
  // std::cout << "BLEN: "  << B_LEN << std::endl;

 // d_h.load(h, 1);
 // d_d.load(d, 1);

   int *d_max_id_val;

    int tid = threadIdx.x;
   // printf("Recursion=%d: Hello World from thread %d block %d\n", iDepth, tid,
     //      blockIdx.x);

    // condition to stop recursive execution
    if (iSize == 1) return;

    // reduce block size to half
    int nthreads = iSize >> 1;

    // thread 0 launches child grid recursively
    if(tid == 0 && nthreads > 0)
    {
       // fill_gpu<<<1, nthreads>>>(1, 1,1,1);
	      for (int i = 1; i <= ((A_LEN + 1) + (B_LEN + 1) - 1); i++) {
    // count++;
    int col_idx = max(0, (i - (B_LEN + 1)));
    int diag_len = min(i, ((A_LEN + 1) - col_idx));

    // launch the kernel: one block by length of diagonal
    int blks = 32;
    if(diag_len / blks >= 1)  {
      dim3 dimBlock(diag_len / blks);
      dim3 dimGrid(blks);
      fill_gpu<<<dimGrid, dimBlock>>>(d_h, d_d, d_seqA, d_seqB, i,
                                    d_max_id_val);
    }
    else {
      dim3 dimBlock(diag_len);
      dim3 dimGrid(1);
      fill_gpu<<<dimGrid, dimBlock>>>(d_h, d_d, d_seqA, d_seqB, i,
                                    d_max_id_val);
    }

        cudaDeviceSynchronize();

       // printf("-------> nested execution depth: %d\n", iDepth);
    }
}}

void smith_water_cpu(Matrixsz h, Matrixsz d, char seqA[], char seqB[]) {

  // populate scoring and direction matrix and find id of max score
  int max_id = fill_cpu(h, d, seqA, seqB);

  // traceback
  std::vector<char> seqA_aligned;
  std::vector<char> seqB_aligned;
  traceback(d, max_id, seqA, seqB, seqA_aligned, seqB_aligned);

  // print aligned sequences
  std::cout << std::endl;
  std::cout << "CPU result: " << std::endl;

  io_seq(seqA_aligned, seqB_aligned);

  // print cpu populated direction and scoring matrix
  io_score(std::string("score.dat"), h, seqA, seqB);
  io_score(std::string("direction.dat"), d, seqA, seqB);
}

void smith_water_gpu(Matrixsz h, Matrixsz d, char seqA[], char seqB[]) {

  std::cout << "GPU result: " << std::endl;

  // allocate and transfer sequence data to device
  char *d_seqA, *d_seqB;
  cudaMalloc(&d_seqA, A_LEN * sizeof(char));
  cudaMalloc(&d_seqB, B_LEN * sizeof(char));
  cudaMemcpy(d_seqA, seqA, A_LEN * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_seqB, seqB, B_LEN * sizeof(char), cudaMemcpyHostToDevice);

  // initialize matrices for gpu
  int Gpu = 1;
  // std::cout << "ALEN: "  << A_LEN << std::endl;
  // std::cout << "BLEN: "  << B_LEN << std::endl;
  Matrixsz d_h(A_LEN + 1, B_LEN + 1, Gpu);
  Matrixsz d_d(A_LEN + 1, B_LEN + 1, Gpu);
 // d_h.load(h, Gpu);
 // d_d.load(d, Gpu);

  // max id and value
  int *d_max_id_val;                   // create pointers and device
  std::vector<int> h_max_id_val(2, 0); // allocate and initialize mem on host
  cudaMalloc(&d_max_id_val, 2 * sizeof(int)); // allocate memory on GPU
  cudaMemcpy(d_max_id_val, h_max_id_val.data(), 2 * sizeof(int),
             cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // loop over diagonals of the matrix
  // int max_threads = 0;
  // int count = 0;
  for (int i = 1; i <= ((A_LEN + 1) + (B_LEN + 1) - 1); i++) {
    // count++;
    int col_idx = max(0, (i - (B_LEN + 1)));
    int diag_len = min(i, ((A_LEN + 1) - col_idx));

    // launch the kernel: one block by length of diagonal
    int blks = 32;
    if(diag_len / blks >= 1)  {
      dim3 dimBlock(diag_len / blks);
      dim3 dimGrid(blks);
      //fill_gpu<<<dimGrid, dimBlock>>>(d_h, d_d, d_seqA, d_seqB, i,
        //                            d_max_id_val);
    }
    else {
      dim3 dimBlock(diag_len);
      dim3 dimGrid(1);
     // fill_gpu<<<dimGrid, dimBlock>>>(d_h, d_d, d_seqA, d_seqB, i,
      //                              d_max_id_val);
    }
    // dim3 dimBlock(diag_len / blks);
    // dim3 dimGrid(blks);
    // if((diag_len / blks)> max_threads)
    //   max_threads = diag_len / blks;
    // // std::cout << "threads: "  << diag_len / blks << std::endl;
    // fill_gpu<<<dimGrid, dimBlock>>>(d_h, d_d, d_seqA, d_seqB, i,
    //                                 d_max_id_val);
    cudaDeviceSynchronize();
  }
  // std::cout << "max threads: "  << max_threads << std::endl;
  // std::cout << "count: "  << count << std::endl;

  // copy data back
  size_t size = (A_LEN + 1) * (B_LEN + 1) * sizeof(float);
  cudaMemcpy(d.elements, d_d.elements, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h.elements, d_h.elements, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_max_id_val.data(), d_max_id_val, 2 * sizeof(int),
             cudaMemcpyDeviceToHost);

  //  std::cout << "   Max score of " << h_max_id_val[1] << " at " <<
  //  max_id_val[0]
  //            << std::endl;

  // traceback
  int max_id = h_max_id_val[0];
  std::vector<char> seqA_aligned;
  std::vector<char> seqB_aligned;
  traceback(d, max_id, seqA, seqB, seqA_aligned, seqB_aligned);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);  

  // visualize output
  io_seq(seqA_aligned, seqB_aligned);
  io_score(std::string("score_gpu.dat"), h, seqA, seqB);
  io_score(std::string("direction_gpu.dat"), d, seqA, seqB);

  std::cout << "   GPU time = " << time << " ms" << std::endl;

  // deallocate memory
  d_h.gpu_deallocate();
  d_d.gpu_deallocate();
  cudaFree(d_seqA);
  cudaFree(d_seqB);
  cudaFree(d_max_id_val);
}

int main() {

  // generate sequences
  char seqA[A_LEN];
  char seqB[B_LEN];
  seq_gen(A_LEN, seqA);
  seq_gen(B_LEN, seqB);

  // print sequences
  std::cout << "Seq A with length " << A_LEN << " is: ";
  for (int i = 0; i < A_LEN; i++)
    std::cout << seqA[i];
  std::cout << std::endl;
  std::cout << "Seq B with length " << B_LEN << " is: ";
  for (int i = 0; i < B_LEN; i++)
    std::cout << seqB[i];
  std::cout << std::endl;

  // initialize scoring and direction matrices
  Matrixsz scr_cpu(A_LEN + 1, B_LEN + 1); // cpu score matrix
  Matrixsz dir_cpu(A_LEN + 1, B_LEN + 1); // cpu direction
  Matrixsz scr_gpu(A_LEN + 1, B_LEN + 1); // gpu score matrix
  Matrixsz dir_gpu(A_LEN + 1, B_LEN + 1); // gpu direction matrix

  // apply initial condition of 0
  for (int i = 0; i < scr_cpu.height; i++) {
    for (int j = 0; j < scr_cpu.width; j++) {
      int id = i * scr_cpu.width + j;
      scr_cpu.elements[id] = 0;
      dir_cpu.elements[id] = 0;
      scr_gpu.elements[id] = 0;
      dir_gpu.elements[id] = 0;
    }
  }

  // visualize initial scoring matrix
  io_score(std::string("init.dat"), scr_cpu, seqA, seqB);

  // CPU
  auto start_cpu = std::chrono::steady_clock::now();
  smith_water_cpu(scr_cpu, dir_cpu, seqA, seqB); // call CPU smith water
  auto end_cpu = std::chrono::steady_clock::now();
  auto diff = end_cpu - start_cpu;
  std::cout << "   CPU time = "
            << std::chrono::duration<double, std::milli>(diff).count() << " ms"
            << std::endl;
  std::cout << std::endl;

  // GPU
  smith_water_gpu(scr_gpu, dir_gpu, seqA, seqB); // call GPU smith water

  // deallocate memory
  scr_cpu.cpu_deallocate();
  dir_cpu.cpu_deallocate();
  scr_gpu.cpu_deallocate();
  dir_gpu.cpu_deallocate();

  return 0;
}
