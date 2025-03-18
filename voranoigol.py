import numpy as np
import networkx as nx
import pygame
from scipy.spatial import Voronoi

# Initialize pygame
pygame.init()
font = pygame.font.Font(None, 18)

# Generate random points
SIZE = 800
num_points = 2000
np.random.seed(400)
points = np.random.rand(num_points, 2) * SIZE
vor = Voronoi(points)

# Create graph and map data
graph = nx.Graph()
point_region_map = {}
region_values = {}
center = np.array([SIZE / 2, SIZE / 2])
min_dist = float('inf')
closest_point_index = None

for i, point in enumerate(points):
    graph.add_node(i, pos=tuple(point))
    region_index = vor.point_region[i]
    point_region_map[region_index] = i
    dist = np.linalg.norm(point - center)
    if dist < min_dist:
        min_dist = dist
        closest_point_index = i

for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
    if v1 != -1 and v2 != -1:
        graph.add_edge(p1, p2)

if closest_point_index is not None:
    closest_region = vor.point_region[closest_point_index]
    region_values[closest_region] = 5.0

# Function to distribute values
def distribute_values(graph, values):
    new_values = {}
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node)) + [node]
        if neighbors:
            share = values.get(node, 0) / len(neighbors)
            for neighbor in neighbors:
                new_values[neighbor] = new_values.get(neighbor, 0) + share
    return new_values

# Screen dimensions
WIDTH, HEIGHT = SIZE, SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Voronoi Diagram")

# Main loop
running = True
while running:
    screen.fill((2, 255, 255))  # Clear screen
    region_values = distribute_values(graph, region_values)
    max_value = max(region_values.values(), default=1)


    for region_index, region in enumerate(vor.regions):
        if -1 in region or len(region) == 0:
            continue
        polygon = [(int(vor.vertices[i][0]), int(vor.vertices[i][1])) for i in region if 0 <= vor.vertices[i][0] < WIDTH and 0 <= vor.vertices[i][1] < HEIGHT]
        if len(polygon) > 2:
            value = region_values.get(point_region_map.get(region_index, None), 0)
            scaled_value = value / max_value if max_value > 0 else 0
            a = max(0, min(255, int(scaled_value * 255)))
            color = (a, a, a)
            pygame.draw.polygon(screen, color, polygon, 0)
            centroid_x = sum(p[0] for p in polygon) / len(polygon)
            centroid_y = sum(p[1] for p in polygon) / len(polygon)
            scaled_value = int(scaled_value*100)
            text = font.render(f'{scaled_value}', True, (0, 0, 0))
            text_rect = text.get_rect(center=(int(centroid_x), int(centroid_y)))
            screen.blit(text, text_rect.topleft)

    for ridge in vor.ridge_vertices:
        if -1 not in ridge:
            v0, v1 = vor.vertices[ridge[0]], vor.vertices[ridge[1]]
            if (0 <= v0[0] < WIDTH and 0 <= v0[1] < HEIGHT) and (0 <= v1[0] < WIDTH and 0 <= v1[1] < HEIGHT):
                pygame.draw.line(screen, (0, 0, 0), (int(v0[0]), int(v0[1])), (int(v1[0]), int(v1[1])), 1)
    for point in points:
        pygame.draw.circle(screen, (255, 155, 125,0.5), (int(point[0]), int(point[1])), 1)
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
