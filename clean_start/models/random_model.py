import time

from clean_start.environments.mini_calico_env import MicroCalicoEnv


NR_OF_TRIES = 1000
average = 0
start_time = time.perf_counter()

for i in range(NR_OF_TRIES):
    env = MicroCalicoEnv()
    env.start_game()
    env.fill_board_randomly()
    score = env.evaluate_board()
    average += score[0]

average /= NR_OF_TRIES
end_time = time.perf_counter()

# 3. Calculate duration
duration = end_time - start_time

print("--------------------------------------------------")
print(f"Average Score: {average}")
print(f"Total Time:    {duration:.4f} seconds")
print(f"Time per Game: {(duration / NR_OF_TRIES) * 1000:.4f} milliseconds")
print("--------------------------------------------------")
