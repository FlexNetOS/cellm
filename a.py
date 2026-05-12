import pygame
from pygame.locals import *

# Initialize Pygame
pygame.init()

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Screen dimensions
screen_width = 800
screen_height = 600

# Create the screen
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Flappy Bird')

# Initialize clock for game frame rate
clock = pygame.time.Clock()

# Game state variables
bird_pos = [100, 50]
bird_angle = 0
pipe_angle = 0
pipe_position = (screen_width // 2, -50)
pipes = []
flap = False

# Bird properties
bird_size = 50
bird_speed = 5
bird_height = 50

# Pipe properties
pipe_size = 30
pipe_width = 10
pipe_height = 100
pipe_gap = 10

# Game variables
game_over = False
score = 0
player_pos = [screen_width // 2, screen_height - bird_size - bird_speed]

def draw_bird():
    font = pygame.font.Font(None, 36)
    text = font.render(str(score), True, WHITE)
    screen.blit(text, (bird_pos[0], bird_pos[1]))

def handle_event(key):
    if key == K_LEFT:
        if pipe_position[0] > 0:
            pipes[0].x -= pipe_speed
        else:
            game_over = True
    elif key == K_RIGHT:
        if pipe_position[0] + pipe_width < screen_width:
            pipes[1].x += pipe_speed
        else:
            game_over = True

def draw_game():
    screen.fill(BLACK)
    for i, (p, g) in enumerate(zip(pipes, pipes)):
        pygame.draw.rect(screen, WHITE, p)
        if game_over and g:
            text = font.render("Game Over", True, WHITE)
            screen.blit(text, (screen_width // 2 - 100, 50))
    draw_bird()
    pygame.display.flip()

def move_pipes():
    global pipe_position
    for p in pipes:
        if p[0] + pipe_gap < screen_width:
            p[0] += pipe_speed
        else:
            game_over = True
            break

if __name__ == "__main__":
    while not game_over:
        clock.tick(60)
        
        event = pygame.event.poll()
        if event.type == KEYDOWN and event.key == K_SPACE or event.type == QUIT:
            game_over = True
        
        draw_game()
        
        # Handle bird movement
        if bird_angle != 0:
            bird_pos[1] += bird_speed * (1 - math.cos(bird_angle / (2 * math.pi)))
        
        # Handle pipe collision with bird
        if player_pos[1] <= bird_pos[1] + bird_size and player_pos[1] >= bird_pos[1] - bird_size:
            if flap:
                bird_pos[1] -= bird_speed
                bird_angle = 0
            else:
                flap = not flap
        
        handle_event(event)
        
        move_pipes()
        
        # Check if the bird has fallen off screen
        if player_pos[1] < bird_pos[1] + bird_size:
            game_over = True

        draw_bird()

    pygame.quit()