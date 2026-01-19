import subprocess
import os
import sys

# Configuration
# Datasets requested by user
datasets = ['movielens']

# Algorithms to run
# Note: 'nle', 'cufe', 'linscan' map to the same underlying implementation in --nodocker mode 
# if they share the same python module, but we run them as separate entries to fulfill the request.
algorithms = ['linscan', 'cufe', 'nle', 'shnsw', 'binary-splitting', 'double-group-testing']

definition_file = 'algos-all-variants.yaml'
results_csv = 'all_results.csv'

def run_command(cmd, msg):
    print(f"\n[INFO] {msg}")
    print(f"Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed: {msg}")
        # Continue to next task despite failure
        pass

def main():
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Use the current python executable to ensure environment consistency
    python_exe = sys.executable

    # 1. Generate Definitions
    print("[INFO] Generating definition file...")
    subprocess.run([python_exe, "create_definitions.py"], check=True)
    
    # 2. Run Benchmarks
    for ds in datasets:
        print(f"\n==========================================")
        print(f" Processing Dataset: {ds}")
        print(f"==========================================")
        
        for algo in algorithms:
            # We use --nodocker because the environment is local
            cmd = [
                python_exe, "-u", "run.py",
                "--dataset", ds,
                "--algorithm", algo,
                "--definitions", definition_file,
                "--count", "10",
                "--nodocker"
            ]
            run_command(cmd, f"Running {algo} on {ds}")

        # 3. Plotting per dataset
        # Note: plot.py scans the 'results/' directory for what has been run.
        # It does NOT take a --definitions argument.
        plot_cmd = [
            python_exe, "plot.py",
            "--dataset", ds,
            "--count", "10",
            "--output", f"results/{ds}_plot.png"
        ]
        run_command(plot_cmd, f"Plotting results for {ds}")

    # 4. Global Data Export
    print(f"\n==========================================")
    print(f" Exporting All Results to {results_csv}")
    print(f"==========================================")
    export_cmd = [
        python_exe, "data_export.py",
        "--output", results_csv
    ]
    run_command(export_cmd, "Exporting CSV")
    print("\n[DONE] Pipeline complete.")

if __name__ == "__main__":
    main()
