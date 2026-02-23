import numpy as np
import pandas as pd

df = pd.DataFrame({'a': range(25378)})
rng = np.random.RandomState(42)

# Simulate Agent A's loop
rng.permutation(10) # arbitrary advance maybe? Wait, actual code:
# Agent A: 
# normal n=2000
normal_df = pd.DataFrame({'a': range(1959772)})
_ = normal_df.sample(n=2000, random_state=rng)
# attacks: dos (5665), fuzzers (21795), exploits (27599)
_ = pd.DataFrame({'a': range(5665)}).sample(n=500, random_state=rng)
_ = pd.DataFrame({'a': range(21795)}).sample(n=500, random_state=rng)
_ = pd.DataFrame({'a': range(27599)}).sample(n=500, random_state=rng)

# Agent B:
# normal n=2000
_ = normal_df.sample(n=2000, random_state=rng)
# attacks: backdoor (1983), shellcode (1511), reconnaissance (13357)
_ = pd.DataFrame({'a': range(1983)}).sample(n=500, random_state=rng)
_ = pd.DataFrame({'a': range(1511)}).sample(n=500, random_state=rng)
_ = pd.DataFrame({'a': range(13357)}).sample(n=500, random_state=rng)

# Agent C:
# normal n=2000
_ = normal_df.sample(n=2000, random_state=rng)
# attacks: worms (171), analysis (2184), generic (25378)
a_c_worms = pd.DataFrame({'a': range(171)}).sample(n=171, random_state=rng)
a_c_analysis = pd.DataFrame({'a': range(2184)}).sample(n=500, random_state=rng)

generic_df = pd.DataFrame({'a': range(25378)})
a_c_generic = generic_df.sample(n=500, random_state=rng)

# --- Test set ---
_ = normal_df.sample(n=2000, random_state=rng)
test_worms = pd.DataFrame({'a': range(171)}).sample(n=171, random_state=rng)
test_analysis = pd.DataFrame({'a': range(2184)}).sample(n=300, random_state=rng)
test_generic = generic_df.sample(n=300, random_state=rng)

overlap_generic = len(set(a_c_generic['a']).intersection(set(test_generic['a'])))
print(f"Generic overlap: {overlap_generic} / {len(test_generic)}")
