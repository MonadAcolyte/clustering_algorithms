print("\n--- Cluster Inspection ---")
for cluster_id in range(1, optimal_k + 1):
    indices = [i for i, label in enumerate(final_labels) if label + 1 == cluster_id]
    print(f"\nCluster {cluster_id} ({len(indices)} sentences):")
    for i in indices[:5]:
        print(f"  - {sentences[i]}")