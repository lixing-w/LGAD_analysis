test_files = ["ACLGAD-test_20.txt", "DCLGAD-test_20.txt"]

import pandas as pd
import matplotlib.pyplot as plt

def plot_iv_curve(file_name):
    """
    Parses the given text file and plots the IV curve with:
    - i_d (Pad)
    - i_gr (GR)
    - Total current (totalCurrent)
    - Sum of pad and guard ring currents (pad + gr)
    The plot is saved as a PNG file with the same name as the input file.
    """
    try:
        # Read the file into a DataFrame, skipping the metadata at the top
        data = pd.read_csv(file_name, skiprows=3)
        
        # Convert the data to absolute values for plotting
        data["voltage"] = data["voltage"].abs()
        data["pad"] = data["pad"].abs()
        data["gr"] = data["gr"].abs()
        data["totalCurrent"] = data["totalCurrent"].abs()
        data["pad_gr_sum"] = data["pad"] + data["gr"]  # Calculate the sum of pad and gr

        # Create the plot
        plt.figure(figsize=(10, 7))
        plt.plot(
            data["voltage"],
            data["pad"],
            label="i_d (Pad)",
            color="blue",
            linestyle="-",
            marker="o",
            linewidth=0.8,
            markersize=3
        )
        plt.plot(
            data["voltage"],
            data["gr"],
            label="i_gr (GR)",
            color="orange",
            linestyle="-",
            marker="o",
            linewidth=0.8,
            markersize=3
        )
        plt.plot(
            data["voltage"],
            data["totalCurrent"],
            label="Total Current",
            color="green",
            linestyle="--",
            marker="s",
            linewidth=0.8,
            markersize=3
        )
        plt.plot(
            data["voltage"],
            data["pad_gr_sum"],
            label="Sum (Pad + GR)",
            color="purple",
            linestyle=":",
            marker="x",
            linewidth=0.8,
            markersize=3
        )

        # Add labels, title, and legend
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.yscale("log")
        plt.title(f"IV Curve - {file_name.split('.')[0]}")
        #plt.ylim(1e-9, 5e-7)  # Set y-axis range
        plt.legend()
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()

        # Save the plot
        output_file_name = file_name.replace(".txt", ".png")
        plt.savefig(output_file_name)
        plt.close()
        print(f"Plot saved as {output_file_name}")
    
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Run the function on the specified files
for test_file in test_files:
    plot_iv_curve(test_file)
