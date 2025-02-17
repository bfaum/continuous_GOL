import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.compiler
import numpy as np
import time
import pygame
from PIL import Image


def resize_and_split_image(image_path, new_width, new_height):
    """
    Resizes the input image to new dimensions and splits it into three numpy arrays
    for the red, green, and blue values of each pixel.

    Args:
    - image_path (str): The path to the image file.
    - new_width (int): The new width of the image.
    - new_height (int): The new height of the image.

    Returns:
    - tuple of numpy.ndarrays: Three arrays for the R, G, B values.
    """
    img = Image.open(image_path)
    img_resized = img.resize((new_width, new_height))
    img_np = np.array(img_resized)
    R = img_np[:, :, 0]
    G = img_np[:, :, 1]
    B = img_np[:, :, 2]
    a=1.0
    return R/255*a, G/255*a, B/255*a


kernel_code = """
__device__ double s(double x) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    return 3*(pow(x,2)) - (2*(pow(x,3)));
}

__device__ double alive_dist(double x) {
    if (x <= 1) return -1;
    if (x < 2) return s(x - 1) - 1;
    if (x <= 3) return 0;
    if (x < 4) return -s(x - 3);
    return -1;
}

__device__ double dead_dist(double x) {
    if (x <= 2) return 0;
    if (x < 3) return s(x-2);
    if (x < 4) return -s(x-3)+1;
    return 0;
}



__global__ void update(const double* __restrict__ board, double* new_board, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        double num_neighbors = 0;

        // Count live neighbors (with boundary checking)
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0) continue;
                int neighbor_x = x + i;
                int neighbor_y = y + j;
                if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
                    int neighbor_idx = neighbor_y * width + neighbor_x;
                    double scale = 1.17157;
                    if (i == 0 && j == 0) scale = 0.828425;
                    num_neighbors += board[neighbor_idx] * scale;
                }
            }
        }
        //apply ccwgol rules        
        double val = board[idx];
        double a = alive_dist(num_neighbors);
        double d = dead_dist(num_neighbors);
        new_board[idx] = .91*val + 1.3*(val * a + (1 - val) * d);
    }
}

__global__ void fill_pixels(double* board, uint8_t* pixels, int board_width, int board_height, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < board_width && y < board_height) {
        int pixel_x_start = x * size;
        int pixel_y_start = y * size;
        double true_color = board[x * board_width + y];
        uint8_t color = (uint8_t)(fmin(fmax(true_color, 0.0), 1.0) * 255);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int pixel_idx = ((pixel_y_start + i) * board_width * size + (pixel_x_start + j)) * 3;
                pixels[pixel_idx] = (uint8_t)fmin((true_color * 255 * 1 + (double)pixels[pixel_idx]*.9),255.0);       // R
                pixels[pixel_idx + 1] = (uint8_t)fmin((true_color * 255 * 1 + (double)pixels[pixel_idx + 1]*.96),255.0);   // G
                pixels[pixel_idx + 2] = (uint8_t)fmin((true_color * 255 * 1 + (double)pixels[pixel_idx + 2]*.0),255.0);   // B
            }
        }
    }
}


"""
#mod = pycuda.compiler.SourceModule(kernel_code, options=["--ptxas-options=-v", "--maxrregcount=24"])

class Board:
    def __init__(self, size, init_arr = None):
        self.height, self.width = size
        if (init_arr is not None):
            self.board = init_arr
        else:
            self.board = np.random.choice([0, 1], size=(self.height, self.width), p=[5/6, 1/6]).astype(np.double)

    def update(self):
        new_board_gpu_result = np.zeros([self.height, self.width], dtype=np.double)
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



pygame.init()
size = 1
N = 1000

image_path = r"me.png"
init_R, init_G, init_B = resize_and_split_image(image_path, N, N)

board = Board((N,N))#, init_G)

screen_width = board.width * size
screen_height = board.height * size
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("ccwgolgpu")

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

    cuda.memcpy_htod(board_gpu, board.board.astype(np.double))

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