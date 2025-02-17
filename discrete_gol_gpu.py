import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler
import numpy as np
import time
import pygame

# Kernel code for matrix multiplication on the GPU
kernel_code = """
__global__ void update(const bool* __restrict__ board, bool* new_board, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int num_neighbors = 0;

        // Count live neighbors (with boundary checking)
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0) continue;
                int neighbor_x = x + i;
                int neighbor_y = y + j;
                if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
                    int neighbor_idx = neighbor_y * width + neighbor_x;
                    num_neighbors += board[neighbor_idx];
                }
            }
        }

        // Apply Conway's Game of Life rules
        if (board[idx] == 1) {
            new_board[idx] = (num_neighbors == 2 || num_neighbors == 3);
        } else {
            new_board[idx] = (num_neighbors == 3);
        }
    }
}

__global__ void fill_pixels(bool* board, uint8_t* pixels, int board_width, int board_height, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < board_width && y < board_height) {
        int pixel_x_start = x * size;
        int pixel_y_start = y * size;
        uint8_t color = board[y * board_width + x] ? 255 : 0;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int pixel_idx = ((pixel_y_start + i) * board_width * size + (pixel_x_start + j)) * 3;
                pixels[pixel_idx] = color;       // R
                pixels[pixel_idx + 1] = color;   // G
                pixels[pixel_idx + 2] = color;   // B
            }
        }
    }
}


"""
glidergun = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
 [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
 [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
 [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
 [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

class Board:
    def __init__(self, size):
        self.height, self.width = size
        self.board = np.random.choice([True, False], size=(self.height, self.width))

        '''
        self.board = np.random.choice([False], size=(self.height, self.width))
        glidergun_bool = np.array(glidergun, dtype=bool)
        h, w = glidergun_bool.shape
        start_y, start_x = (0, 0)
        self.board[start_y:start_y + h, start_x:start_x + w] = glidergun_bool
        start_y, start_x = (300, 300)
        self.board[start_y:start_y + h, start_x:start_x + w] = glidergun_bool
        '''
    def update(self):
        new_board_gpu_result = np.zeros([self.height, self.width], dtype=bool)
        new_board_gpu = cuda.mem_alloc(self.board.nbytes)
        board_gpu = cuda.mem_alloc(self.board.nbytes)

        cuda.memcpy_htod(board_gpu, self.board)

        block_size = 16
        grid_size_x = int(np.ceil(self.width / block_size))
        grid_size_y = int(np.ceil(self.height / block_size))

        #do the thing here
        update(board_gpu,
               new_board_gpu,
               np.int32(self.width),
               np.int32(self.height),
               block=(block_size, block_size, 1), grid=(grid_size_x, grid_size_y))

        #return to cpu
        cuda.memcpy_dtoh(new_board_gpu_result, new_board_gpu)
        self.board = new_board_gpu_result


# Compile the kernel code
mod = pycuda.compiler.SourceModule(kernel_code)
update = mod.get_function("update")
fill_pixels = mod.get_function("fill_pixels")

#test
pygame.init()
size = 1
N = 1000
board = Board((N,N))
screen_width = board.width * size
screen_height = board.height * size
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("cwgolgpu")

surface = pygame.Surface((screen_width, screen_height))

# Allocate GPU memory
board_gpu = cuda.mem_alloc(board.board.nbytes)
pixels_gpu = cuda.mem_alloc(screen_width * screen_height * 3)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update the board
    update_start_time = time.time()
    board.update()
    update_end_time = time.time()
    update_time_ms = (update_end_time - update_start_time) * 1000
    print(f"Update time: {update_time_ms:.2f} ms")

    render_start_time = time.time()

    cuda.memcpy_htod(board_gpu, board.board)

    block_size = (16, 16, 1)
    grid_size = (int(np.ceil(board.width / 16)), int(np.ceil(board.height / 16)), 1)

    fill_pixels(board_gpu, pixels_gpu, np.int32(board.width), np.int32(board.height), np.int32(size), block=block_size, grid=grid_size)

    pixels = np.empty((screen_height, screen_width, 3), dtype=np.uint8)
    cuda.memcpy_dtoh(pixels, pixels_gpu)

    pygame.surfarray.blit_array(surface, pixels)
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    render_end_time = time.time()
    render_time_ms = (render_end_time - render_start_time) * 1000
    print(f"Render time: {render_time_ms:.2f} ms")
    pygame.time.delay(0)

board_gpu.free()
pixels_gpu.free()
pygame.quit()