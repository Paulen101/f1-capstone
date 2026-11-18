"""TUM's compound-choice network on its own, for our pit-timing network's stops (.venv-tum, needs TensorFlow).

TUM's supervised VSE has two networks: nn_supervised_tirechange (pit or stay) and
nn_supervised_compoundchoice (which compound when it pits). Our network replaces the first; this module runs
the second exactly as racesim/src/vse_supervised.py does (same six inputs, same preprocessor, same rule for
walking down the compound ranking), with one difference: it is always told that one tyre change remains,
because our network makes one stop. TUM's own VSE takes that number from its basic strategy, which can plan
two stops (Austin). With one change left TUM's rule never mounts the current compound again, so the race
stays legal (two compounds).

The TUM source is not changed; the network and preprocessor are loaded from TUM's input files.
"""
import pickle
from typing import Optional

import numpy as np


class TumCompoundChoice:
    def __init__(self, vse_paths: dict, location: str, available_compounds: list, param_dry_compounds: list):
        import tensorflow as tf
        with open(vse_paths["supervised_preprocessor_cc"], "rb") as stream:
            self.preprocessor = pickle.load(stream)
        if location not in self.preprocessor.cat_dict["location"]:
            raise ValueError("TUM's compound-choice network does not know %s" % location)
        self.interpreter = tf.lite.Interpreter(model_path=vse_paths["supervised_nnmodel_cc"])
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        self.location = location
        self.avail = [c for c in available_compounds if c not in ("I", "W")]   # vse.py: avail_dry_compounds
        self.param = list(param_dry_compounds)

    def rel_compound(self, compound: str) -> int:
        """vse.py: rel_compound_num_curlap."""
        return self.avail.index(compound) + (1 if len(self.avail) == 2 else 0)

    def probabilities(self, cur_compound: str, raceprogress_prevlap: float, used_2compounds: bool = False,
                      remaining_changes: int = 1) -> np.ndarray:
        """Network output for (hardest, middle, softest), the inputs of vse_supervised.preprocess_features."""
        x = np.zeros((1, 6))
        x[:, 0] = raceprogress_prevlap
        x[:, 1] = self.preprocessor.transform_cat_dict(X_cat_str=[self.location], featurename="location")
        x[:, 2] = self.rel_compound(cur_compound)
        x[:, 3] = remaining_changes
        x[:, 4] = used_2compounds
        x[:, 5] = len(self.avail)
        self.interpreter.set_tensor(self.input_index, self.preprocessor.transform(x, dtype_out=np.float32))
        self.interpreter.invoke()
        return np.array(self.interpreter.get_tensor(self.output_index)[0], dtype=np.float64)

    def choose(self, cur_compound: str, raceprogress_prevlap: float) -> Optional[str]:
        """Compound for a stop decided now: vse_supervised.make_decision's ranking walk, one change left."""
        for rel in np.argsort(-self.probabilities(cur_compound, raceprogress_prevlap)):
            if len(self.avail) == 2:
                rel -= 1
                if rel < 0:
                    continue
            compound = self.avail[rel]
            if compound not in self.param or compound == cur_compound:
                continue
            return compound
        return None
