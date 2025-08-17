#include <stdio.h>
#include <cuda_runtime.h>

// Kernel de CUDA (se ejecuta en la GPU)
__global__ void sumaVectores(float* A, float* B, float* C, int n) {
    int i = threadIdx.x;  // Obtiene el índice del hilo actual
    if (i < n) {
        C[i] = A[i] + B[i];  // Realiza la suma elemento por elemento
    }
}

int main() {
    const int N = 5;  // Tamaño de los vectores
    float h_A[N], h_B[N], h_C[N];  // Vectores en el host (CPU)
    float *d_A, *d_B, *d_C;       // Punteros para device (GPU)

    // Inicializar vectores de entrada
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // 1. Reservar memoria en la GPU
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // 2. Copiar datos del host al device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Lanzar el kernel con N hilos (1 bloque de N hilos)
    sumaVectores<<<1, N>>>(d_A, d_B, d_C, N);

    // 4. Copiar resultado de vuelta al host
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 5. Liberar memoria de la GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Mostrar resultados
    printf("Vector A: ");
    for (int i = 0; i < N; i++) printf("%.2f  ", h_A[i]);
    
    printf("\nVector B: ");
    for (int i = 0; i < N; i++) printf("%.2f  ", h_B[i]);
    
    printf("\nResultado: ");
    for (int i = 0; i < N; i++) printf("%.2f  ", h_C[i]);

    return 0;
}
