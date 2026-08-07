import random
import numpy as np
import logging


class BalancedMaterialPositionSplit:

    def __init__(
        self,
        n_positions,
        n_materials,
        n_val_pos,
        n_test_pos,
        random_seed=32
    ):

        self.n_positions = n_positions
        self.n_materials = n_materials
        self.n_val_pos = n_val_pos
        self.n_test_pos = n_test_pos
        self.random_seed = random_seed


    def split(self, reader):

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        train_combos = set()
        val_combos = set()
        test_combos = set()


        for material in range(self.n_materials):

            positions = list(range(self.n_positions))
            random.shuffle(positions)

            test_positions = positions[:self.n_test_pos]

            val_positions = positions[
                self.n_test_pos:
                self.n_test_pos + self.n_val_pos
            ]

            train_positions = positions[
                self.n_test_pos + self.n_val_pos:
            ]


            test_combos |= {
                (material, p)
                for p in test_positions
            }

            val_combos |= {
                (material, p)
                for p in val_positions
            }

            train_combos |= {
                (material, p)
                for p in train_positions
            }


        train = []
        val = []
        test = []


        for name in reader.list_samples():

            metadata = reader.get_metadata(name)

            combo = (
                metadata["material_id"],
                metadata["position"]
            )

            if combo in test_combos:
                test.append(name)

            elif combo in val_combos:
                val.append(name)

            elif combo in train_combos:
                train.append(name)


        logging.info(
            f"Train={len(train)}, Val={len(val)}, Test={len(test)}"
        )

        return train, val, test