
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

import sys
sys.stdout.reconfigure(encoding='utf-8')

# Clean up
if os.path.exists("knowledge_base.csv"): os.remove("knowledge_base.csv")

# Create data
print("🧪 Generating Test Data...")
df_cls = pd.DataFrame(np.random.randn(200, 10), columns=[f"feat_{i}" for i in range(10)])
# Make a target that is actually learnable so we can see improvement
# y = (feat_0 > 0) & (feat_1 > 0)
y = ((df_cls['feat_0'] + df_cls['feat_1']) > 0).astype(int)
df_cls['target'] = y
df_cls.to_csv("evo_test.csv", index=False)

acc_history = []
print("🔄 Simulating 15 Evolutionary Generations...")

for i in range(15):
    # Run pipeline quiet
    ret = os.system(f"python pipeline.py evo_test.csv --target target --epochs 5 > run_{i}.log 2>&1")
    if ret != 0:
        print(f"❌ Run {i} Falied.")
        break
        
    # Extract metric
    # pipeline.py prints: "   ✓ Run Performance (Accuracy): 0.xxxx"
    # But it also stores in knowledge_base.csv. Let's read it from there.
    try:
        kb = pd.read_csv("knowledge_base.csv")
        last_metric = kb.iloc[-1]['final_metric']
        acc_history.append(last_metric)
        print(f"   Gen {i+1}: Accuracy = {last_metric:.4f}")
    except:
        print(f"   Gen {i+1}: Failed to read metric.")

print("\n📈 Evolutionary Trajectory:")
print(acc_history)

# Simple validation: Did we improve or stay stable high?
# Note: Initial runs might vary due to exploration.
avg_first_3 = np.mean(acc_history[:3])
avg_last_3 = np.mean(acc_history[-3:])

print(f"\nStart Avg: {avg_first_3:.4f}")
print(f"End Avg:   {avg_last_3:.4f}")

if avg_last_3 >= avg_first_3:
    print("✅ System shows non-decreasing performance trend (or potential improvement).")
else:
    print("⚠️ Performance degraded. (Might be due to noise/randomness in small sample).")

# Cleanup
# os.remove("evo_test.csv")
