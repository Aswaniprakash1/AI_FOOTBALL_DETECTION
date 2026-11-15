
import numpy as np


class FootballStatistics:
    def __init__(self, fps=30):
        self.fps = fps
        self.player_positions = {}
        self.distances = {}

    def add_position(self, player_id, position, frame_number):
        if position is None:
            return

        player_id = int(player_id)
        position = np.array(position, dtype=np.float32)

        if player_id not in self.player_positions:
            self.player_positions[player_id] = []

        self.player_positions[player_id].append(
            (frame_number, position)
        )

        positions = self.player_positions[player_id]

        if len(positions) >= 2:
            previous = positions[-2][1]
            current = positions[-1][1]

            distance = np.linalg.norm(current - previous)

            self.distances[player_id] = (
                self.distances.get(player_id, 0) + distance
            )

    def get_player_distance(self, player_id):
        return self.distances.get(int(player_id), 0)

    def get_all_distances(self):
        return self.distances.copy()

    def reset(self):
        self.player_positions.clear()
        self.distances.clear()
