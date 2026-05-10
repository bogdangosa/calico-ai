import pandas as pd
import os
from datetime import datetime
import secrets
import string

def generate_short_id(length=8):
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def save_simulation_details(game_data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(game_data)
    df.to_csv(path, index=False)


def append_to_results(results_data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame([results_data])

    file_exists = os.path.isfile(path)
    df.to_csv(path, mode='a', index=False, header=not file_exists)